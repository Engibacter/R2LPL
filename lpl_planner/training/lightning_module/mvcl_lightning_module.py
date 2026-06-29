import collections.abc as cabc
import logging
import re
from typing import Any, Dict, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from lpl_planner.planning.scene.scene_feature.features import AnchorIndice, ReplayPlannerTargets

logger = logging.getLogger(__name__)


class MVCLLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        cfg: Optional[DictConfig] = None,
        learning_rate: float = 2e-4,
        warmup_steps: int = 2000,
        check_invalid_grad: bool = False,
        continual_method: str = "er",
        replay_loss_weight: float = 1.0,
        derpp_distill_weight: float = 0.5,
        derpp_temperature: float = 1.0,
        derpp_rank_margin: float = 0.5,
        prediction_distill_weight: float = 0.0,
        distill_source: str = "stored_logits",
        expert_loss_weight: float = 0.0,
        mas_enabled: bool = False,
        mas_lambda: float = 0.0,
        mas_update_alpha: float = 0.5,
        mas_max_batches: int = 32,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.check_invalid_grad = check_invalid_grad
        self.continual_method = continual_method.strip().lower()
        self.replay_loss_weight = float(replay_loss_weight)
        self.derpp_distill_weight = float(derpp_distill_weight)
        self.derpp_temperature = max(float(derpp_temperature), 1e-6)
        self.derpp_rank_margin = float(derpp_rank_margin)
        self.prediction_distill_weight = max(float(prediction_distill_weight), 0.0)
        self.distill_source = str(distill_source or "stored_logits").strip().lower()
        self.expert_loss_weight = max(float(expert_loss_weight), 0.0)
        object.__setattr__(self, "_teacher_model", None)
        self.mas_enabled = bool(mas_enabled)
        self.mas_lambda = float(mas_lambda)
        self.mas_update_alpha = min(max(float(mas_update_alpha), 0.0), 1.0)
        self.mas_max_batches = max(int(mas_max_batches), 1)
        self.automatic_optimization = False
        self.active_task_name: Optional[str] = None
        self.active_task_index: Optional[int] = None
        self._task_train_loss_sum: Dict[str, float] = {}
        self._task_train_count: Dict[str, int] = {}
        self._token_best_val_loss: Dict[str, float] = {}
        self._token_last_val_loss: Dict[str, float] = {}
        self._mas_omega: Dict[str, torch.Tensor] = {}
        self._mas_reference_params: Dict[str, torch.Tensor] = {}
        self._validation_sources: Dict[int, str] = {0: "val"}

    def _task_metric_prefix(self) -> Optional[str]:
        if self.active_task_name is None:
            return None
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(self.active_task_name)).strip("_") or "task"
        safe_index = -1 if self.active_task_index is None else int(self.active_task_index)
        return f"tasks/{safe_index:02d}_{safe_name}"

    def _log_metric(self, name: str, value: Any, include_task_prefix: bool = True, **kwargs) -> None:
        self.log(name, value, **kwargs)
        task_prefix = self._task_metric_prefix()
        if include_task_prefix and task_prefix is not None:
            self.log(f"{task_prefix}/{name}", value, **kwargs)

    def set_task_context(self, task_name: str, task_index: int) -> None:
        self.active_task_name = task_name
        self.active_task_index = int(task_index)
        self._task_train_loss_sum = {}
        self._task_train_count = {}

    def set_validation_sources(self, source_names) -> None:
        if not source_names:
            self._validation_sources = {0: "val"}
            return
        self._validation_sources = {idx: str(name) for idx, name in enumerate(source_names)}

    def set_teacher_model(self, teacher_model) -> None:
        object.__setattr__(self, "_teacher_model", teacher_model)
        if self._teacher_model is None:
            return
        self._teacher_model.eval()
        for param in self._teacher_model.parameters():
            param.requires_grad_(False)

    def pop_task_statistics(self) -> Dict[str, Dict[str, float]]:
        importance_scores = {}
        for token, total_loss in self._task_train_loss_sum.items():
            count = max(self._task_train_count.get(token, 1), 1)
            importance_scores[token] = total_loss / count

        forgetting_scores = {}
        for token, last_loss in self._token_last_val_loss.items():
            best_loss = self._token_best_val_loss.get(token, last_loss)
            forgetting_scores[token] = max(0.0, float(last_loss - best_loss))

        return {
            "importance": importance_scores,
            "forgetting": forgetting_scores,
        }

    def _reduce_loss_value(self, value):
        if torch.is_tensor(value) and value.ndim > 0:
            return value.mean()
        return value

    def _extract_sample_losses(self, loss_dict: Dict[str, Any], key: str = "total_loss") -> Optional[torch.Tensor]:
        value = loss_dict.get(key)
        if value is None or (not torch.is_tensor(value)):
            return None
        if value.ndim == 0:
            return value.unsqueeze(0)
        if value.ndim == 1:
            return value.detach().float()
        return value.reshape(value.shape[0], -1).mean(dim=-1).detach().float()

    def _log_loss_dict(self, prefix: str, loss_dict: Dict[str, Any], batch_size: int) -> None:
        for key, value in loss_dict.items():
            if value is None:
                continue
            reduced_value = self._reduce_loss_value(value)
            self._log_metric(f"{prefix}/{key}", reduced_value, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)

    def _move_to_device(self, obj, device):
        if hasattr(obj, "to_device") and callable(getattr(obj, "to_device")):
            return obj.to_device(device)
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, cabc.Mapping):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            moved = [self._move_to_device(v, device) for v in obj]
            return type(obj)(moved) if not isinstance(obj, list) else moved
        return obj

    def _match_param_tensor(self, tensor: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        if tensor.device == param.device and tensor.dtype == param.dtype:
            return tensor
        return tensor.to(device=param.device, dtype=param.dtype)

    def _ensure_mas_state_device_alignment(self) -> None:
        if not self._mas_omega and not self._mas_reference_params:
            return

        for name, param in self.model.named_parameters():
            omega = self._mas_omega.get(name)
            if omega is not None:
                self._mas_omega[name] = self._match_param_tensor(omega, param)

            ref_param = self._mas_reference_params.get(name)
            if ref_param is not None:
                self._mas_reference_params[name] = self._match_param_tensor(ref_param, param)

    def _extract_tokens(self, features: Dict[str, Any]) -> Tuple[str, ...]:
        token_feature = features.get("scenario_token")
        if token_feature is None:
            return tuple()
        token_value = getattr(token_feature, "token", token_feature)
        if isinstance(token_value, list):
            return tuple(str(token) for token in token_value)
        return (str(token_value),)

    def _record_train_statistics(self, tokens: Tuple[str, ...], sample_losses: Optional[torch.Tensor]) -> None:
        if not tokens or sample_losses is None or sample_losses.numel() == 0:
            return
        if sample_losses.numel() == 1 and len(tokens) > 1:
            sample_losses = sample_losses.repeat(len(tokens))
        if sample_losses.numel() != len(tokens):
            logger.warning("train sample loss count %s does not match token count %s", sample_losses.numel(), len(tokens))
            return
        for token, sample_loss in zip(tokens, sample_losses.tolist()):
            self._task_train_loss_sum[token] = self._task_train_loss_sum.get(token, 0.0) + float(sample_loss)
            self._task_train_count[token] = self._task_train_count.get(token, 0) + 1

    def _record_validation_statistics(self, tokens: Tuple[str, ...], sample_losses: Optional[torch.Tensor]) -> None:
        if not tokens or sample_losses is None or sample_losses.numel() == 0:
            return
        if sample_losses.numel() == 1 and len(tokens) > 1:
            sample_losses = sample_losses.repeat(len(tokens))
        if sample_losses.numel() != len(tokens):
            logger.warning("val sample loss count %s does not match token count %s", sample_losses.numel(), len(tokens))
            return
        for token, sample_loss in zip(tokens, sample_losses.tolist()):
            scalar_loss = float(sample_loss)
            self._token_last_val_loss[token] = scalar_loss
            best = self._token_best_val_loss.get(token)
            if best is None or scalar_loss < best:
                self._token_best_val_loss[token] = scalar_loss

    def _forward_batch(self, batch: Tuple[Dict[str, Any], Dict[str, Any]], prefix: str, log_metrics: bool = True):
        features, targets = batch
        device = self.device
        features = self._move_to_device(features, device)
        targets = self._move_to_device(targets, device)
        batch_size = features["scene_feature"].ego_feature.ego_current_state.shape[0]
        prediction = self.model.forward(features, targets)
        loss_dict = prediction["loss_dict"]
        total_loss = self._reduce_loss_value(loss_dict["total_loss"])
        if log_metrics:
            self._log_loss_dict(prefix, loss_dict, batch_size)
            if "mean_expert_distance" in prediction:
                self._log_metric(f"{prefix}/mean_dist", prediction["mean_expert_distance"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
        return total_loss, prediction, loss_dict, features, targets, batch_size

    def _extract_distillation_logits(self, prediction: Dict[str, Any], targets: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        pred_scores = prediction.get("anchor_scores")
        if pred_scores is None:
            pred_scores = prediction.get("candidate_scores")

        target_scores = targets.get("anchor_scores")
        if target_scores is not None and hasattr(target_scores, "aggregated_scores"):
            target_scores = target_scores.aggregated_scores

        indices = prediction.get("indices")
        if pred_scores is None or target_scores is None:
            return None, None

        if hasattr(pred_scores, "aggregated_scores"):
            pred_scores = pred_scores.aggregated_scores

        if indices is not None and torch.is_tensor(indices) and torch.is_tensor(target_scores):
            if target_scores.dim() == 2 and pred_scores.dim() == 2 and target_scores.shape[1] != pred_scores.shape[1]:
                target_scores = target_scores.gather(1, indices.long())

        if not (torch.is_tensor(pred_scores) and torch.is_tensor(target_scores)):
            return None, None
        if pred_scores.shape != target_scores.shape:
            return None, None
        return pred_scores.float(), target_scores.float()

    def _resolve_anchor_indices(self, targets: Dict[str, Any]) -> Optional[torch.Tensor]:
        anchor_indices = targets.get("anchor_indice")
        if anchor_indices is None:
            return None
        if hasattr(anchor_indices, "indice"):
            anchor_indices = anchor_indices.indice
        elif hasattr(anchor_indices, "data") and torch.is_tensor(anchor_indices.data):
            anchor_indices = anchor_indices.data
        if not torch.is_tensor(anchor_indices):
            return None
        if anchor_indices.dim() == 1:
            anchor_indices = anchor_indices.unsqueeze(-1)
        return anchor_indices.long()

    def _resolve_anchor_scores(self, targets: Dict[str, Any]) -> Optional[torch.Tensor]:
        anchor_scores = targets.get("anchor_scores")
        if anchor_scores is None:
            return None
        if hasattr(anchor_scores, "aggregated_scores"):
            anchor_scores = anchor_scores.aggregated_scores
        elif hasattr(anchor_scores, "data") and torch.is_tensor(anchor_scores.data):
            anchor_scores = anchor_scores.data
        if not torch.is_tensor(anchor_scores):
            return None
        return anchor_scores.float()

    def _extract_replay_planner_targets(self, targets: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        replay_targets = targets.get(ReplayPlannerTargets.get_feature_unique_name())
        if replay_targets is None:
            return None, None

        replay_indices = getattr(replay_targets, "anchor_indices", None)
        teacher_logits = getattr(replay_targets, "teacher_logits", None)
        if not (torch.is_tensor(replay_indices) and torch.is_tensor(teacher_logits)):
            return None, None
        if replay_indices.dim() == 1:
            replay_indices = replay_indices.unsqueeze(0)
        if teacher_logits.dim() == 1:
            teacher_logits = teacher_logits.unsqueeze(0)
        if replay_indices.shape != teacher_logits.shape:
            return None, None
        return replay_indices.long(), teacher_logits.float()

    def _resolve_local_teacher_index(
        self,
        replay_indices: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        anchor_indices = targets.get(AnchorIndice.get_feature_unique_name())
        teacher_indices = self._resolve_anchor_indices(targets) if anchor_indices is not None else None
        if teacher_indices is None:
            return teacher_logits.argmax(dim=-1)

        teacher_mask = (replay_indices.unsqueeze(-1) == teacher_indices.unsqueeze(1)).any(dim=-1)
        fallback_idx = teacher_logits.argmax(dim=-1)
        if not teacher_mask.any():
            return fallback_idx

        masked_teacher_logits = teacher_logits.masked_fill(~teacher_mask, float("-inf"))
        matched_idx = masked_teacher_logits.argmax(dim=-1)
        has_match = teacher_mask.any(dim=-1)
        return torch.where(has_match, matched_idx, fallback_idx)

    def _compute_derpp_stored_logit_loss(
        self,
        features: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        replay_indices, teacher_logits = self._extract_replay_planner_targets(targets)
        if replay_indices is None or teacher_logits is None:
            return None

        subset_out = self.model.score_candidate_subset(
            scene_features=features["scene_feature"],
            anchor_indices=replay_indices.to(device=self.device),
            no_grad=False,
            return_prediction=False,
        )
        logits = subset_out["candidate_scores"].float()
        teacher_logits = teacher_logits.to(device=logits.device, dtype=logits.dtype)
        local_teacher_idx = self._resolve_local_teacher_index(
            replay_indices=replay_indices.to(device=logits.device, dtype=torch.long),
            teacher_logits=teacher_logits,
            targets=targets,
        )
        if local_teacher_idx is None:
            return None

        temperature = self.derpp_temperature
        pred_log_prob = F.log_softmax(logits / temperature, dim=-1)
        teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
        kl_loss = F.kl_div(pred_log_prob, teacher_prob, reduction="batchmean") * (temperature ** 2)
        ce_loss = F.cross_entropy(logits, local_teacher_idx, reduction="mean")

        negative_mask = torch.ones_like(logits, dtype=torch.bool)
        negative_mask.scatter_(1, local_teacher_idx.unsqueeze(1), False)
        rank_loss = torch.tensor(0.0, device=logits.device)
        valid_rank_rows = negative_mask.any(dim=-1)
        if valid_rank_rows.any():
            safe_negative_mask = negative_mask.clone()
            safe_negative_mask[~valid_rank_rows] = True
            hard_neg_idx = teacher_logits.masked_fill(~safe_negative_mask, float("-inf")).argmax(dim=-1)
            pos_logit = logits.gather(1, local_teacher_idx.unsqueeze(1)).squeeze(1)
            neg_logit = logits.gather(1, hard_neg_idx.unsqueeze(1)).squeeze(1)
            rank_term = F.relu(self.derpp_rank_margin - (pos_logit - neg_logit))
            rank_loss = rank_term[valid_rank_rows].mean()

        total = ce_loss + kl_loss + rank_loss
        return {"total": total, "ce": ce_loss, "kl": kl_loss, "rank": rank_loss}

    def _compute_derpp_anchor_score_loss(self, prediction: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        logits = prediction.get("candidate_scores")
        sampled_indices = prediction.get("indices")
        if logits is None or sampled_indices is None or (not torch.is_tensor(logits)) or (not torch.is_tensor(sampled_indices)):
            zero = torch.tensor(0.0, device=self.device)
            return {"total": zero, "ce": zero, "kl": zero, "rank": zero}

        logits = logits.float()
        sampled_indices = sampled_indices.long()
        teacher_indices = self._resolve_anchor_indices(targets)
        teacher_scores = self._resolve_anchor_scores(targets)

        sampled_teacher_scores = None
        if teacher_scores is not None:
            if teacher_scores.dim() == 2 and teacher_scores.shape[1] >= sampled_indices.max().item() + 1:
                sampled_teacher_scores = teacher_scores.gather(1, sampled_indices)
            elif teacher_scores.shape == logits.shape:
                sampled_teacher_scores = teacher_scores

        teacher_mask = None
        local_teacher_idx = None
        if teacher_indices is not None:
            teacher_indices = teacher_indices.to(device=sampled_indices.device, dtype=torch.long)
            teacher_mask = (sampled_indices.unsqueeze(-1) == teacher_indices.unsqueeze(1)).any(dim=-1)
            if teacher_mask.any():
                if sampled_teacher_scores is not None:
                    teacher_scores_masked = sampled_teacher_scores.masked_fill(~teacher_mask, float("-inf"))
                    local_teacher_idx = teacher_scores_masked.argmax(dim=-1)
                else:
                    teacher_mask_int = teacher_mask.to(dtype=torch.int64)
                    local_teacher_idx = teacher_mask_int.argmax(dim=-1)
                valid_teacher_rows = teacher_mask.any(dim=-1)
            else:
                valid_teacher_rows = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
        else:
            valid_teacher_rows = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)

        if sampled_teacher_scores is not None:
            positive_mask = sampled_teacher_scores > 0
        else:
            positive_mask = torch.zeros_like(logits, dtype=torch.bool)

        if local_teacher_idx is None and positive_mask.any():
            local_teacher_idx = sampled_teacher_scores.argmax(dim=-1)
            valid_teacher_rows = positive_mask.any(dim=-1)

        ce_loss = torch.tensor(0.0, device=self.device)
        if local_teacher_idx is not None and valid_teacher_rows.any():
            ce_loss = F.cross_entropy(logits[valid_teacher_rows], local_teacher_idx[valid_teacher_rows], reduction="mean")

        kl_loss = torch.tensor(0.0, device=self.device)
        valid_kl_rows = positive_mask.any(dim=-1)
        if sampled_teacher_scores is not None and valid_kl_rows.any():
            target_scores = sampled_teacher_scores[valid_kl_rows].clamp_min(0.0)
            target_prob = target_scores / target_scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            pred_log_prob = F.log_softmax(logits[valid_kl_rows] / self.derpp_temperature, dim=-1)
            kl_loss = F.kl_div(pred_log_prob, target_prob, reduction="batchmean") * (self.derpp_temperature ** 2)

        rank_loss = torch.tensor(0.0, device=self.device)
        if local_teacher_idx is not None:
            negative_mask = torch.ones_like(logits, dtype=torch.bool)
            negative_mask.scatter_(1, local_teacher_idx.unsqueeze(1), False)
            if sampled_teacher_scores is not None:
                negative_mask = negative_mask & (sampled_teacher_scores <= 0)
            valid_rank_rows = valid_teacher_rows & negative_mask.any(dim=-1)
            if valid_rank_rows.any():
                neg_logits = logits.masked_fill(~negative_mask, float("-inf"))
                hard_neg_idx = neg_logits.argmax(dim=-1)
                pos_logit = logits.gather(1, local_teacher_idx.unsqueeze(1)).squeeze(1)
                neg_logit = logits.gather(1, hard_neg_idx.unsqueeze(1)).squeeze(1)
                rank_gap = pos_logit - neg_logit
                rank_loss = F.relu(self.derpp_rank_margin - rank_gap)[valid_rank_rows].mean()

        total = ce_loss + kl_loss + rank_loss
        return {"total": total, "ce": ce_loss, "kl": kl_loss, "rank": rank_loss}

    def _compute_derpp_distillation_loss(
        self,
        features: Dict[str, Any],
        prediction: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        if self.distill_source in {"teacher_model", "model", "previous_model"} and self._teacher_model is not None:
            return self._compute_teacher_model_distillation_loss(features, prediction)
        stored_logit_terms = self._compute_derpp_stored_logit_loss(features, targets)
        if stored_logit_terms is not None:
            return stored_logit_terms
        return self._compute_derpp_anchor_score_loss(prediction, targets)

    def _zero_distill_terms(self) -> Dict[str, torch.Tensor]:
        zero = torch.tensor(0.0, device=self.device)
        return {"total": zero, "ce": zero, "kl": zero, "rank": zero, "prediction": zero}

    def _compute_prediction_distillation_loss(
        self,
        student_out: Dict[str, Any],
        teacher_out: Dict[str, Any],
    ) -> torch.Tensor:
        pred_loss = torch.tensor(0.0, device=self.device)
        student_modes = student_out.get("agent_prediction_modes")
        teacher_modes = teacher_out.get("agent_prediction_modes")
        if torch.is_tensor(student_modes) and torch.is_tensor(teacher_modes) and student_modes.shape == teacher_modes.shape:
            pred_loss = pred_loss + F.mse_loss(student_modes.float(), teacher_modes.to(student_modes.device).float())

        student_conf = student_out.get("agent_prediction_confidence")
        teacher_conf = teacher_out.get("agent_prediction_confidence")
        if torch.is_tensor(student_conf) and torch.is_tensor(teacher_conf) and student_conf.shape == teacher_conf.shape:
            temperature = self.derpp_temperature
            pred_log_prob = F.log_softmax(student_conf.float() / temperature, dim=-1)
            teacher_prob = F.softmax(teacher_conf.to(student_conf.device).float() / temperature, dim=-1)
            pred_loss = pred_loss + F.kl_div(pred_log_prob, teacher_prob, reduction="batchmean") * (temperature ** 2)
        return pred_loss

    def _compute_teacher_model_distillation_loss(
        self,
        features: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        if self._teacher_model is None or not hasattr(self.model, "score_candidate_subset"):
            return self._zero_distill_terms()
        self._teacher_model.to(self.device)
        self._teacher_model.eval()

        replay_indices = prediction.get("indices")
        if replay_indices is None or not torch.is_tensor(replay_indices):
            return self._zero_distill_terms()
        replay_indices = replay_indices.to(device=self.device, dtype=torch.long)

        student_out = self.model.score_candidate_subset(
            scene_features=features["scene_feature"],
            anchor_indices=replay_indices,
            no_grad=False,
            return_prediction=self.prediction_distill_weight > 0.0,
        )
        with torch.no_grad():
            teacher_out = self._teacher_model.score_candidate_subset(
                scene_features=features["scene_feature"],
                anchor_indices=replay_indices,
                no_grad=True,
                return_prediction=self.prediction_distill_weight > 0.0,
            )

        student_logits = student_out["candidate_scores"].float()
        teacher_logits = teacher_out["candidate_scores"].to(device=student_logits.device, dtype=student_logits.dtype)
        temperature = self.derpp_temperature
        pred_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
        kl_loss = F.kl_div(pred_log_prob, teacher_prob, reduction="batchmean") * (temperature ** 2)
        local_teacher_idx = teacher_logits.argmax(dim=-1)
        ce_loss = F.cross_entropy(student_logits, local_teacher_idx, reduction="mean")

        negative_mask = torch.ones_like(student_logits, dtype=torch.bool)
        negative_mask.scatter_(1, local_teacher_idx.unsqueeze(1), False)
        rank_loss = torch.tensor(0.0, device=student_logits.device)
        if negative_mask.any(dim=-1).any():
            hard_neg_idx = teacher_logits.masked_fill(~negative_mask, float("-inf")).argmax(dim=-1)
            pos_logit = student_logits.gather(1, local_teacher_idx.unsqueeze(1)).squeeze(1)
            neg_logit = student_logits.gather(1, hard_neg_idx.unsqueeze(1)).squeeze(1)
            rank_loss = F.relu(self.derpp_rank_margin - (pos_logit - neg_logit)).mean()

        prediction_loss = torch.tensor(0.0, device=student_logits.device)
        if self.prediction_distill_weight > 0.0:
            prediction_loss = self._compute_prediction_distillation_loss(student_out, teacher_out)

        total = ce_loss + kl_loss + rank_loss + self.prediction_distill_weight * prediction_loss
        return {"total": total, "ce": ce_loss, "kl": kl_loss, "rank": rank_loss, "prediction": prediction_loss}

    def _compute_mas_regularization_loss(self) -> torch.Tensor:
        if (not self.mas_enabled) or self.mas_lambda <= 0.0 or not self._mas_omega or not self._mas_reference_params:
            return torch.tensor(0.0, device=self.device)

        self._ensure_mas_state_device_alignment()
        reg_loss = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            omega = self._mas_omega.get(name)
            ref_param = self._mas_reference_params.get(name)
            if omega is None or ref_param is None:
                continue
            reg_loss = reg_loss + (omega * (param - ref_param).pow(2)).sum()
        return self.mas_lambda * reg_loss

    @torch.no_grad()
    def has_mas_state(self) -> bool:
        return bool(self._mas_omega) and bool(self._mas_reference_params)

    def update_mas_state(self, dataloader, max_batches: Optional[int] = None) -> None:
        if not self.mas_enabled:
            return

        max_batches = max_batches or self.mas_max_batches
        was_training = self.model.training
        self.model.eval()
        current_omega = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        batch_count = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            self.model.zero_grad(set_to_none=True)
            features, targets = batch
            features = self._move_to_device(features, self.device)
            targets = self._move_to_device(targets, self.device)
            prediction = self.model.forward(features, targets)
            candidate_scores = prediction.get("candidate_scores")
            if candidate_scores is not None and torch.is_tensor(candidate_scores):
                mas_objective = 0.5 * candidate_scores.float().pow(2).sum(dim=-1).mean()
            else:
                planned_traj = prediction.get("trajectory")
                if planned_traj is None or (not torch.is_tensor(planned_traj)):
                    continue
                mas_objective = 0.5 * planned_traj.float().pow(2).sum(dim=(-1, -2)).mean()

            mas_objective.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in current_omega:
                    current_omega[name] += param.grad.detach().abs()
            batch_count += 1

        if batch_count <= 0:
            self.model.train(was_training)
            return

        for name in current_omega:
            current_omega[name] = current_omega[name] / float(batch_count)
            if name in self._mas_omega:
                previous_omega = self._match_param_tensor(self._mas_omega[name], current_omega[name])
                self._mas_omega[name] = self.mas_update_alpha * previous_omega + (1.0 - self.mas_update_alpha) * current_omega[name]
            else:
                self._mas_omega[name] = current_omega[name]

        self._mas_reference_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self.model.zero_grad(set_to_none=True)
        self.model.train(was_training)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["mas_state"] = {
            "omega": {name: tensor.detach().cpu() for name, tensor in self._mas_omega.items()},
            "reference_params": {name: tensor.detach().cpu() for name, tensor in self._mas_reference_params.items()},
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        mas_state = checkpoint.get("mas_state") or {}
        omega_state = mas_state.get("omega") or {}
        reference_state = mas_state.get("reference_params") or {}
        self._mas_omega = {
            name: tensor.detach().clone()
            for name, tensor in omega_state.items()
            if torch.is_tensor(tensor)
        }
        self._mas_reference_params = {
            name: tensor.detach().clone()
            for name, tensor in reference_state.items()
            if torch.is_tensor(tensor)
        }

    def _flatten_grads(self) -> torch.Tensor:
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros(param.numel(), device=self.device, dtype=torch.float32))
            else:
                grads.append(param.grad.detach().reshape(-1).float().clone())
        return torch.cat(grads) if grads else torch.zeros(1, device=self.device)

    def _assign_flattened_grads(self, flat_grad: torch.Tensor) -> None:
        offset = 0
        for param in self.model.parameters():
            numel = param.numel()
            grad_slice = flat_grad[offset: offset + numel].view_as(param)
            if param.grad is None:
                param.grad = grad_slice.clone()
            else:
                param.grad.copy_(grad_slice)
            offset += numel

    def _step_scheduler(self) -> None:
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return
        if isinstance(schedulers, list):
            for scheduler in schedulers:
                scheduler.step()
        else:
            schedulers.step()

    def training_step(self, batch, batch_idx: int):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        if isinstance(batch, dict):
            current_batch = batch.get("current")
            memory_batch = batch.get("memory")
            expert_batch = batch.get("expert")
        else:
            current_batch = batch
            memory_batch = None
            expert_batch = None

        if current_batch is None:
            raise ValueError("current batch is required for continual training")

        if self.continual_method == "agem" and memory_batch is not None:
            memory_loss, _, memory_loss_dict, memory_features, _, memory_batch_size = self._forward_batch(memory_batch, "train/memory", log_metrics=True)
            self.manual_backward(memory_loss)
            ref_grad = self._flatten_grads()
            optimizer.zero_grad()

            current_loss, _, current_loss_dict, current_features, _, current_batch_size = self._forward_batch(current_batch, "train/current", log_metrics=True)
            self.manual_backward(current_loss)
            current_grad = self._flatten_grads()
            dot = torch.dot(current_grad, ref_grad)
            ref_norm = torch.dot(ref_grad, ref_grad)
            if dot < 0 and ref_norm > 0:
                projected_grad = current_grad - (dot / ref_norm) * ref_grad
                self._assign_flattened_grads(projected_grad)
                self._log_metric("train/agem_projected", torch.tensor(1.0, device=self.device), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            else:
                self._log_metric("train/agem_projected", torch.tensor(0.0, device=self.device), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

            mas_loss = self._compute_mas_regularization_loss()
            total_loss = current_loss + mas_loss
            self._log_metric("train/current_total_loss", current_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=current_batch_size)
            self._log_metric("train/memory_total_loss", memory_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=memory_batch_size)
            if self.mas_enabled:
                self._log_metric("train/mas_loss", mas_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self._record_train_statistics(self._extract_tokens(current_features), self._extract_sample_losses(current_loss_dict))
        else:
            current_loss, _, current_loss_dict, current_features, _, current_batch_size = self._forward_batch(current_batch, "train/current", log_metrics=True)
            total_loss = current_loss
            self._log_metric("train/current_total_loss", current_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=current_batch_size)
            self._record_train_statistics(self._extract_tokens(current_features), self._extract_sample_losses(current_loss_dict))

            if memory_batch is not None and self.continual_method in {"er", "derpp", "der++"}:
                memory_loss, memory_prediction, memory_loss_dict, memory_features, memory_targets, memory_batch_size = self._forward_batch(memory_batch, "train/memory", log_metrics=True)
                weighted_memory_loss = self.replay_loss_weight * memory_loss
                total_loss = total_loss + weighted_memory_loss
                self._log_metric("train/memory_total_loss", memory_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=memory_batch_size)
                self._log_metric("train/memory_weighted_loss", weighted_memory_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=memory_batch_size)

                if self.continual_method in {"derpp", "der++"}:
                    distill_terms = self._compute_derpp_distillation_loss(memory_features, memory_prediction, memory_targets)
                    weighted_distill = self.derpp_distill_weight * distill_terms["total"]
                    total_loss = total_loss + weighted_distill
                    self._log_metric("train/derpp_distill_loss", distill_terms["total"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=memory_batch_size)
                    self._log_metric("train/derpp_ce_loss", distill_terms["ce"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=memory_batch_size)
                    self._log_metric("train/derpp_kl_loss", distill_terms["kl"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=memory_batch_size)
                    self._log_metric("train/derpp_rank_loss", distill_terms["rank"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=memory_batch_size)
                    if "prediction" in distill_terms:
                        self._log_metric("train/prediction_distill_loss", distill_terms["prediction"], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=memory_batch_size)

            if expert_batch is not None and self.expert_loss_weight > 0.0:
                expert_loss, _, expert_loss_dict, expert_features, _, expert_batch_size = self._forward_batch(expert_batch, "train/expert", log_metrics=True)
                weighted_expert_loss = self.expert_loss_weight * expert_loss
                total_loss = total_loss + weighted_expert_loss
                self._log_metric("train/expert_total_loss", expert_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=expert_batch_size)
                self._log_metric("train/expert_weighted_loss", weighted_expert_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=expert_batch_size)
                self._record_train_statistics(self._extract_tokens(expert_features), self._extract_sample_losses(expert_loss_dict))

            mas_loss = self._compute_mas_regularization_loss()
            total_loss = total_loss + mas_loss
            if self.mas_enabled:
                self._log_metric("train/mas_loss", mas_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

            self.manual_backward(total_loss)

        optimizer.step()
        self._step_scheduler()

        lr = optimizer.param_groups[0]["lr"]
        self._log_metric("learning_rate", torch.tensor(lr, device=self.device), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self._log_metric("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss.detach()

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        if isinstance(batch, dict):
            batch = batch.get("current") or batch.get("memory")
        source_name = self._validation_sources.get(dataloader_idx, f"val_{dataloader_idx}")
        prefix = "val" if source_name == "val" else f"val/{source_name}"
        val_loss, _, loss_dict, features, _, batch_size = self._forward_batch(batch, prefix, log_metrics=True)
        self._record_validation_statistics(self._extract_tokens(features), self._extract_sample_losses(loss_dict))
        return val_loss

    def on_after_backward(self):
        if self.check_invalid_grad or getattr(self.model, "debug", False):
            for name, param in self.model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    logger.warning("Gradient for %s contains NaN or Inf values!", name)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-2)

        if hasattr(self.trainer, "estimated_stepping_batches") and self.trainer.estimated_stepping_batches:
            total_steps = int(self.trainer.estimated_stepping_batches)
        elif self.trainer.max_steps and self.trainer.max_steps > 0:
            total_steps = int(self.trainer.max_steps)
        else:
            total_steps = 100000
        warmup_steps = max(self.warmup_steps, int(0.03 * total_steps))
        epoch_step = total_steps // max(int(self.trainer.max_epochs), 1)
        warmup_steps = max(warmup_steps, epoch_step + 10)
        self.warmup_steps = warmup_steps

        warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-5)
        seq = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
        return [optimizer], [{"scheduler": seq, "interval": "step", "frequency": 1}]
