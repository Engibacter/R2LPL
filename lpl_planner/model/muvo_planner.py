from collections import OrderedDict
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


from lpl_planner.model.modules.scene_encoder import SceneFeatureIDX, SceneStateEncoder
from lpl_planner.planning.scene.scene_feature.features import SceneFeature, AgentPrediction, AnchorIndice, AnchorScores
from lpl_planner.model.modules import Mlp, LayerNorm
from lpl_planner.planning.scene.trajectory_library import TrajectoryState
from lpl_planner.model.modules.utils import (
    VectorizedFourierEncoding,
    select_best_anchors_by_expert_shape,
    score_anchors_by_expert_traj,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
LOGIT_CLAMP = 30.0

POSE_IDX = [TrajectoryState.X,
           TrajectoryState.Y,
           TrajectoryState.HEADING]

ALL_IDX = [TrajectoryState.X,
           TrajectoryState.Y,
           TrajectoryState.HEADING,
           TrajectoryState.VELOCITY_X,
           TrajectoryState.ACCELERATION_X,
           TrajectoryState.YAW_RATE]


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.float().unsqueeze(-1)
    denom = weight.sum(dim=1).clamp_min(1.0)
    return (x * weight).sum(dim=1) / denom

    
class MUVOPlanner(nn.Module):
    def __init__(self,
                 intermidiate_dim: int,
                 future_sampling: TrajectorySampling,
                 encoder_depth: int = 2,
                 planning_decoder_depths: int = 8,
                 prediction_decoder_depths: int = 5,
                 use_agent_context_decoder: bool = False,
                 agent_context_decoder_depths: int = 2,
                 planner_anchor_path: str = 'planner_anchor.npy',
                 regression_loss_weight: float = 1,
                 regression_yaw_loss_weight: float = 0.2,
                 classification_loss_weight: float = 8,
                 train_anchor_num: int = 256,
                 test_anchor_num: int = 256,
                 use_prediction: bool = True,
                 prediction_loss_weight: float = 1.0,
                 score_chunk_size: int = 256,
                 imitation_target_tau: float = 0.05,   # Smaller values sharpen the target; 1.0 approximates the legacy scale.
                 prediction_time_weight_gamma: float = 0.5,
                 teacher_anchor_ratio: float = 0.2,
                 teacher_min_anchor_num: int = 1,
                 teacher_prefilter_topk: int = 128,
                 num_prediction_modes: int = 3,
                 prediction_confidence_loss_weight: float = 0.1,
                 prediction_use_cv_delta: bool = False,
                 prediction_delta_scale: float = 1.0,
                 prediction_ade_weight: float = 1.0,
                 prediction_fde_weight: float = 2.0,
                 prediction_yaw_weight: float = 0.5,
                 prediction_yaw_end_weight: float = 0.5,
                 prediction_vel_weight: float = 0.2,
                 prediction_vel_end_weight: float = 0.2,
                 prediction_diversity_weight: float = 0.0,
                 prediction_diversity_margin: float = 2.0,
                 teacher_ce_weight: float = 1.0,
                 teacher_ce_label_smoothing: float = 0.1,
                 anchor_score_kl_weight: float = 1.0,
                 use_anchor_score_kl_loss: bool = True,
                 anchor_score_neg_loss_weight: float = 1.0,
                 neg_delta_margin: float = 0.5,
                 neg_delta_floor: float = 1.0,
                 neg_delta_slope: float = 1.0,
                 use_expert_distance_guidance: bool = True,
                 debug: bool = False,
                 ):
        super().__init__()
        self.debug = bool(debug)
        self.use_prediction = bool(use_prediction)
        self.use_agent_context_decoder = bool(use_agent_context_decoder)
        if self.use_prediction and self.use_agent_context_decoder:
            raise ValueError("use_agent_context_decoder and use_prediction are mutually exclusive in MUVOPlanner.")
        self.state_encoder = SceneStateEncoder(unified_dim=intermidiate_dim, 
                                               encoder_depth=encoder_depth,
                                               debug=self.debug)

        self.imitation_target_tau = imitation_target_tau
        self.regression_loss_weight = regression_loss_weight
        self.regression_yaw_loss_weight = max(float(regression_yaw_loss_weight), 0.0)
        self.classification_loss_weight = classification_loss_weight
        self.prediction_loss_weight = prediction_loss_weight

        # forward settings
        self.train_anchor_num = train_anchor_num
        self.test_anchor_num = test_anchor_num
        self.use_prediction = bool(use_prediction)

        # loss and evaluation settings
        self.prediction_time_weight_gamma = prediction_time_weight_gamma
        self.teacher_anchor_ratio = float(teacher_anchor_ratio)
        self.teacher_min_anchor_num = max(int(teacher_min_anchor_num), 0)
        self.teacher_prefilter_topk = max(int(teacher_prefilter_topk), 1)
        self.num_prediction_modes = max(int(num_prediction_modes), 1)
        self.prediction_confidence_loss_weight = float(prediction_confidence_loss_weight)
        self.prediction_use_cv_delta = bool(prediction_use_cv_delta)
        self.prediction_delta_scale = float(prediction_delta_scale)
        self.prediction_ade_weight = float(prediction_ade_weight)
        self.prediction_fde_weight = float(prediction_fde_weight)
        self.prediction_yaw_weight = float(prediction_yaw_weight)
        self.prediction_yaw_end_weight = float(prediction_yaw_end_weight)
        self.prediction_vel_weight = float(prediction_vel_weight)
        self.prediction_vel_end_weight = float(prediction_vel_end_weight)
        self.prediction_diversity_weight = max(float(prediction_diversity_weight), 0.0)
        self.prediction_diversity_margin = max(float(prediction_diversity_margin), 0.0)
        self.teacher_ce_weight = max(float(teacher_ce_weight), 0.0)
        self.teacher_ce_label_smoothing = min(max(float(teacher_ce_label_smoothing), 0.0), 1.0)
        self.anchor_score_kl_weight = max(float(anchor_score_kl_weight), 0.0)
        self.use_anchor_score_kl_loss = bool(use_anchor_score_kl_loss)
        self.anchor_score_neg_loss_weight = max(float(anchor_score_neg_loss_weight), 0.0)
        self.neg_delta_margin = float(neg_delta_margin)
        self.neg_delta_floor = max(float(neg_delta_floor), 1e-3)
        self.neg_delta_slope = max(float(neg_delta_slope), 1e-3)
        self.use_expert_distance_guidance = bool(use_expert_distance_guidance)
        self.score_chunk_size = score_chunk_size

        planner_anchor = np.load(planner_anchor_path)
        planner_anchor = planner_anchor[..., ALL_IDX]
        if planner_anchor.shape[1] < future_sampling.num_poses:
            raise ValueError(
                f"planner anchors are too short for single-step planning: "
                f"anchor_horizon={planner_anchor.shape[1]}, requested={future_sampling.num_poses}"
            )
        planner_anchor = planner_anchor[:, :future_sampling.num_poses]

        self.planner_anchor_path = planner_anchor_path
        self.planner_anchor = nn.Parameter(
            torch.tensor(planner_anchor, dtype=torch.float32),
            requires_grad=False,
        )

        self.num_poses = future_sampling.num_poses
        self.anchor_horizon = self.planner_anchor.shape[1]
        self.num_steps = 1
        self.time_step = future_sampling.interval_length
        self.future_sampling = future_sampling
        self.use_prediction = bool(use_prediction)
        self.prediction_steps = future_sampling.num_poses

        self.anchor_encoder = AnchorEncoder(
            self.planner_anchor[..., :3],
            d_model=intermidiate_dim,
            encoding_depth=10,
        )
        
        self.planning_decoder = nn.ModuleList([
            PlanningDecoderLayer(d_model=intermidiate_dim)
            for _ in range(planning_decoder_depths)
        ])
        
        self.ego_query_pre_branch = nn.Sequential(
            nn.Linear(intermidiate_dim, intermidiate_dim * 2),
            LayerNorm(intermidiate_dim * 2),
            nn.ReLU(),
            nn.Linear(intermidiate_dim * 2, intermidiate_dim),
        )

        self.planning_scorer = Mlp(
            intermidiate_dim,
            intermidiate_dim * 4,
            1,
        )

        if self.use_prediction:
            self.prediction_decoder = nn.ModuleList([
                PredictionDecoderLayer(d_model=intermidiate_dim, nhead=8)
                for _ in range(prediction_decoder_depths)
            ])
            
            self.prediction_head = PredictionHead(
                intermidiate_dim,
                self.prediction_steps,
                num_modes=self.num_prediction_modes,
                dt=self.time_step,
                use_cv_delta_prediction=self.prediction_use_cv_delta,
                delta_scale=self.prediction_delta_scale,
            )
        elif self.use_agent_context_decoder:
            self.agent_context_decoder = nn.ModuleList([
                PredictionDecoderLayer(d_model=intermidiate_dim, nhead=8)
                for _ in range(agent_context_decoder_depths)
            ])

        self._configure_debug_modules()

    def _configure_debug_modules(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, Mlp):
                module.debug_checks = self.debug
                module.debug_sanitize = self.debug
                module.debug_clamp = 1e6
                module.debug_name = name

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        adjusted_state_dict = OrderedDict(state_dict)

        if self.use_agent_context_decoder and not self.use_prediction:
            has_context_weights = any(key.startswith("agent_context_decoder.") for key in adjusted_state_dict)
            if not has_context_weights:
                for key, value in list(adjusted_state_dict.items()):
                    if key.startswith("prediction_decoder."):
                        context_key = key.replace("prediction_decoder.", "agent_context_decoder.", 1)
                        adjusted_state_dict[context_key] = value

        if not self.use_prediction:
            adjusted_state_dict = OrderedDict(
                (key, value)
                for key, value in adjusted_state_dict.items()
                if not key.startswith("prediction_decoder.") and not key.startswith("prediction_head.")
            )
            current_keys = set(self.state_dict().keys())
            adjusted_state_dict = OrderedDict(
                (key, value)
                for key, value in adjusted_state_dict.items()
                if (not key.startswith("agent_context_decoder.")) or key in current_keys
            )

        try:
            return super().load_state_dict(adjusted_state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(adjusted_state_dict, strict=strict)

    def _debug_tensor(
        self,
        tensor: Optional[torch.Tensor],
        name: str,
        clamp: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        if (not self.debug) or tensor is None or (not torch.is_tensor(tensor)):
            return tensor

        if tensor.dtype == torch.bool:
            return tensor

        finite = torch.isfinite(tensor)
        if not finite.all():
            num_bad = int((~finite).sum().item())
            logger.warning("Tensor %s has %s non-finite values. Sanitizing with nan_to_num.", name, num_bad)
            limit = LOGIT_CLAMP if clamp is None else clamp
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=limit, neginf=-limit)

        if clamp is not None and torch.is_floating_point(tensor):
            tensor = torch.clamp(tensor, -clamp, clamp)
        return tensor


    def _build_decoder_context(
        self,
        encoded_state: torch.Tensor,
        state_mask: torch.Tensor,
        state_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        

        ego_feat = encoded_state[:, -1:, :]  # (B, 1, D)
        num_agent = int((state_ids[0] == SceneFeatureIDX.AGENT).sum().item())
        
        if num_agent == 0:
            agent_feat = torch.empty(
                encoded_state.size(0),
                0,
                encoded_state.size(2),
                device=encoded_state.device,
                dtype=encoded_state.dtype,
            )
            agent_mask = torch.empty(
                encoded_state.size(0),
                0,
                device=encoded_state.device,
                dtype=torch.bool,
            )
        else:  
            agent_feat = encoded_state[:, -1-num_agent:-1, :]  # (B, num_agent, D)
            agent_mask = state_mask[:, -1-num_agent:-1]  # (B, num_agent)

        env_feat = encoded_state[:, :-(1+num_agent), :]  # (B, num_env, D)
        env_mask = state_mask[:, :-(1+num_agent)]  # (B, num_env)

        return {
            'ego_feat': ego_feat,
            'agent_feat': agent_feat,
            'agent_mask': agent_mask,
            'env_feat': env_feat,
            'env_mask': env_mask,
            'scene_feat': encoded_state,
            'scene_mask': state_mask,
        }

    def _encode_scene_context(self, scene_feature: SceneFeature) -> Dict[str, torch.Tensor]:
        
        encoded_state, state_mask, state_ids = self.state_encoder(scene_feature)
        encoded_state = self._debug_tensor(encoded_state, "encoded_state", clamp=1e6)
        decoder_context = self._build_decoder_context(encoded_state, state_mask, state_ids)
        for key, value in decoder_context.items():
            if torch.is_tensor(value):
                decoder_context[key] = self._debug_tensor(value, f"decoder_context.{key}", clamp=1e6)
        return decoder_context

    def _run_agent_context_decoder(self, decoder_context: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if (not self.use_agent_context_decoder) or (not hasattr(self, "agent_context_decoder")):
            return decoder_context
        if decoder_context['agent_feat'].size(1) == 0:
            return decoder_context

        updated_context = dict(decoder_context)
        for block in self.agent_context_decoder:
            updated_context['agent_feat'] = block(
                agent_feat=updated_context['agent_feat'],
                agent_mask=updated_context['agent_mask'],
                scene_feat=updated_context['scene_feat'],
                scene_mask=updated_context['scene_mask'],
            )
            updated_context['agent_feat'] = self._debug_tensor(
                updated_context['agent_feat'],
                "agent_feat_after_agent_context_decoder_block",
                clamp=1e6,
            )
        return updated_context

    def _select_top1_prediction_mode(
        self,
        agent_pred_trajs: torch.Tensor,
        agent_pred_confidence: torch.Tensor,
    ) -> torch.Tensor:
        top1_idx = agent_pred_confidence.argmax(dim=-1)
        return agent_pred_trajs.gather(
            2,
            top1_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                -1,
                -1,
                1,
                agent_pred_trajs.shape[3],
                agent_pred_trajs.shape[4],
            ),
        ).squeeze(2)


    def _run_prediction_modules(
        self,
        scene_feature: SceneFeature,
        decoder_context: Dict[str, torch.Tensor],
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if not self.use_prediction:
            return decoder_context, None, None, None, None
        if not (hasattr(self, "prediction_decoder") and hasattr(self, "prediction_head")):
            raise RuntimeError("use_prediction=True requires prediction_decoder and prediction_head to be initialized.")

        updated_context = dict(decoder_context)
        for block in self.prediction_decoder:
            updated_context['agent_feat'] = block(
                agent_feat=updated_context['agent_feat'],
                agent_mask=updated_context['agent_mask'],
                scene_feat=updated_context['scene_feat'],
                scene_mask=updated_context['scene_mask'],
            )
            updated_context['agent_feat'] = self._debug_tensor(
                updated_context['agent_feat'],
                "agent_feat_after_prediction_decoder_block",
                clamp=1e6,
            )

        agent_pred_trajs, agent_pred_confidence = self.prediction_head(
            updated_context['agent_feat'],
            agent_current_state=scene_feature.agent_feature.agent_current_state,
            agent_valid_mask=scene_feature.agent_feature.agent_mask,
        )
        agent_pred_trajs = self._debug_tensor(agent_pred_trajs, "agent_pred_trajs", clamp=1e6)
        agent_pred_confidence = self._debug_tensor(agent_pred_confidence, "agent_pred_confidence", clamp=1e3)
        top1_agent_pred = self._select_top1_prediction_mode(
            agent_pred_trajs,
            agent_pred_confidence,
        )
        top1_agent_pred = self._debug_tensor(top1_agent_pred, "top1_agent_pred", clamp=1e6)
        agent_future_mask = updated_context['agent_mask'].unsqueeze(-1).expand(
            updated_context['agent_mask'].shape[0],
            top1_agent_pred.size(1),
            self.prediction_steps,
        )
        return updated_context, agent_pred_trajs, agent_pred_confidence, top1_agent_pred, agent_future_mask

    def _run_query_decoder(
        self,
        traj_query: torch.Tensor,
        decoder_context: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        for block in self.planning_decoder:
            traj_query = block(
                traj_feature=traj_query,
                ego_feature=decoder_context['ego_feat'],
                agent_feature=decoder_context['agent_feat'],
                agent_mask=decoder_context['agent_mask'],
                env_feature=decoder_context['env_feat'],
                env_mask=decoder_context['env_mask'],
            )
            traj_query = self._debug_tensor(traj_query, "traj_query_after_decoder_block", clamp=1e6)
        return traj_query

    def _get_teacher_anchor_sample_num(self, candidate_num: int) -> int:
        teacher_sample_num = int(round(candidate_num * max(self.teacher_anchor_ratio, 0.0)))
        teacher_sample_num = max(teacher_sample_num, self.teacher_min_anchor_num)
        return min(max(teacher_sample_num, 1), candidate_num)

    def _resolve_teacher_indices_from_targets(
        self,
        anchor_indices: Optional[AnchorIndice],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if anchor_indices is None:
            return None

        if hasattr(anchor_indices, 'indice'):
            teacher_indices = anchor_indices.indice
        elif hasattr(anchor_indices, 'data') and torch.is_tensor(anchor_indices.data):
            teacher_indices = anchor_indices.data
        elif torch.is_tensor(anchor_indices):
            teacher_indices = anchor_indices
        else:
            raise TypeError(f"unsupported anchor_indices type: {type(anchor_indices)!r}")

        teacher_indices = teacher_indices.to(device=device, dtype=torch.long)
        if teacher_indices.shape[0] != batch_size:
            raise ValueError(
                f"anchor_indices batch mismatch: {teacher_indices.shape[0]} vs {batch_size}"
            )

        if teacher_indices.dim() == 1:
            teacher_indices = teacher_indices.unsqueeze(-1)
        else:
            teacher_indices = teacher_indices.reshape(batch_size, -1)

        if teacher_indices.shape[1] == 0:
            return None

        if self.num_steps == 1 and teacher_indices.shape[1] > 1:
            teacher_indices = teacher_indices[:, :1]

        return teacher_indices

    def _resolve_anchor_scores_from_targets(
        self,
        anchor_scores: Optional[AnchorScores],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if anchor_scores is None:
            return None

        if hasattr(anchor_scores, 'aggregated_scores'):
            resolved_scores = anchor_scores.aggregated_scores
        elif hasattr(anchor_scores, 'data') and torch.is_tensor(anchor_scores.data):
            resolved_scores = anchor_scores.data
        elif torch.is_tensor(anchor_scores):
            resolved_scores = anchor_scores
        else:
            raise TypeError(f"unsupported anchor_scores type: {type(anchor_scores)!r}")

        resolved_scores = resolved_scores.to(device=device, dtype=torch.float32)
        if resolved_scores.shape[0] != batch_size:
            raise ValueError(
                f"anchor_scores batch mismatch: {resolved_scores.shape[0]} vs {batch_size}"
            )

        if resolved_scores.dim() == 1:
            resolved_scores = resolved_scores.unsqueeze(0)
        elif resolved_scores.dim() > 2:
            resolved_scores = resolved_scores.reshape(batch_size, -1)

        return torch.nan_to_num(resolved_scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    def _preselect_teacher_anchor_indices_from_expert(
        self,
        expert_anchor_target: torch.Tensor,
        teacher_sample_num: int,
    ) -> Optional[torch.Tensor]:
        if teacher_sample_num <= 0:
            return None

        anchor_bank = self.planner_anchor.to(
            device=expert_anchor_target.device,
            dtype=expert_anchor_target.dtype,
        )
        coarse_pool_num = max(self.teacher_prefilter_topk, teacher_sample_num)
        selected_indices, _ = select_best_anchors_by_expert_shape(
            expert_anchor_target.float(),
            anchor_bank,
            preselect_k=coarse_pool_num,
            topk=teacher_sample_num,
        )
        return selected_indices.long()

    def _select_teacher_anchor_index_from_full_bank_by_expert_dist(
        self,
        expert_anchor_target: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = expert_anchor_target.shape[0]
        anchor_bank = self.planner_anchor.to(
            device=expert_anchor_target.device,
            dtype=torch.float32,
        )
        expert_anchor_target = expert_anchor_target.float()
        best_score = torch.full(
            (batch_size, 1),
            float('inf'),
            device=expert_anchor_target.device,
            dtype=torch.float32,
        )
        best_index = torch.zeros(
            batch_size,
            1,
            dtype=torch.long,
            device=expert_anchor_target.device,
        )
        chunk_size = max(1, int(self.score_chunk_size))

        for start in range(0, anchor_bank.shape[0], chunk_size):
            end = min(anchor_bank.shape[0], start + chunk_size)
            anchor_chunk = anchor_bank[start:end].unsqueeze(0).expand(batch_size, -1, -1, -1)
            chunk_score = score_anchors_by_expert_traj(
                expert_anchor_target,
                anchor_chunk,
            ).float()
            chunk_best_score, chunk_local_index = chunk_score.min(dim=-1, keepdim=True)
            update_mask = chunk_best_score < best_score
            best_score = torch.where(update_mask, chunk_best_score, best_score)
            best_index = torch.where(update_mask, chunk_local_index.long() + start, best_index)

        return best_index

    def _compose_teacher_indices(
        self,
        target_teacher_indices: Optional[torch.Tensor],
        expert_anchor_target: torch.Tensor,
        candidate_num: int,
    ) -> Optional[torch.Tensor]:
        teacher_sample_num = self._get_teacher_anchor_sample_num(candidate_num)
        if target_teacher_indices is None:
            return self._preselect_teacher_anchor_indices_from_expert(
                expert_anchor_target,
                teacher_sample_num,
            )

        target_teacher_num = target_teacher_indices.shape[1]
        extra_teacher_num = max(teacher_sample_num - target_teacher_num, 0)
        if extra_teacher_num <= 0:
            return target_teacher_indices

        expert_teacher_indices = self._preselect_teacher_anchor_indices_from_expert(
            expert_anchor_target,
            extra_teacher_num,
        )
        if expert_teacher_indices is None:
            return target_teacher_indices

        return torch.cat([target_teacher_indices, expert_teacher_indices], dim=1)


    def _sample_candidate_indices(
        self,
        dynamic_scores: torch.Tensor,
        total_num: int,
        teacher_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, total_anchor_num = dynamic_scores.shape
        if total_num >= total_anchor_num:
            return torch.arange(total_anchor_num, device=dynamic_scores.device).unsqueeze(0).expand(batch_size, -1)

        used_index = torch.empty(batch_size, total_num, dtype=torch.long, device=dynamic_scores.device)
        for batch_idx in range(batch_size):
            forced: List[int] = []
            if teacher_indices is not None:
                seen = set()
                for idx in teacher_indices[batch_idx].tolist():
                    idx = int(idx)
                    if idx in seen:
                        continue
                    seen.add(idx)
                    forced.append(idx)
                    if len(forced) >= total_num:
                        break

            forced_tensor = torch.tensor(forced, device=dynamic_scores.device, dtype=torch.long)
            remaining = total_num - forced_tensor.numel()
            if remaining > 0:
                sample_prob = dynamic_scores[batch_idx].clone()
                if forced_tensor.numel() > 0:
                    sample_prob[forced_tensor] = 0.0
                if sample_prob.sum() <= 0:
                    sample_prob = torch.ones_like(sample_prob)
                    if forced_tensor.numel() > 0:
                        sample_prob[forced_tensor] = 0.0
                sample_prob = sample_prob / sample_prob.sum().clamp_min(1e-12)
                sampled = torch.multinomial(sample_prob, num_samples=remaining, replacement=False)
                selected = torch.cat([forced_tensor, sampled], dim=0)
            else:
                selected = forced_tensor[:total_num]

            # Randomize local candidate order after teacher injection so the teacher does
            # not always occupy the first slot, while still remaining inside the sampled set.
            if selected.numel() > 1:
                permutation = torch.randperm(selected.numel(), device=dynamic_scores.device)
                selected = selected[permutation]

            used_index[batch_idx] = selected
        return used_index

    def _sample_candidate_indices_with_anchor_scores(
        self,
        anchor_scores: torch.Tensor,
        total_num: int,
        teacher_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, total_anchor_num = anchor_scores.shape
        if total_num >= total_anchor_num:
            return torch.arange(total_anchor_num, device=anchor_scores.device).unsqueeze(0).expand(batch_size, -1)

        teacher_sample_num = self._get_teacher_anchor_sample_num(total_num)

        def _sample_mask_from_pool(pool_mask: torch.Tensor, sample_counts: torch.Tensor) -> torch.Tensor:
            sampled_mask = torch.zeros_like(pool_mask)
            max_pick = int(sample_counts.max().item())
            if max_pick <= 0:
                return sampled_mask

            random_key = torch.rand(pool_mask.shape, device=pool_mask.device, dtype=anchor_scores.dtype)
            random_key = random_key.masked_fill(~pool_mask, 2.0)
            sampled_idx = torch.topk(random_key, k=max_pick, dim=1, largest=False).indices
            valid_pick = torch.arange(max_pick, device=pool_mask.device).unsqueeze(0) < sample_counts.unsqueeze(1)
            sampled_mask.scatter_(1, sampled_idx, valid_pick)
            return sampled_mask & pool_mask

        selected_mask = torch.zeros(batch_size, total_anchor_num, dtype=torch.bool, device=anchor_scores.device)

        if teacher_indices is not None:
            valid_teacher = (teacher_indices >= 0) & (teacher_indices < total_anchor_num)
            safe_teacher_indices = teacher_indices.masked_fill(~valid_teacher, 0)
            selected_mask.scatter_(1, safe_teacher_indices.long(), valid_teacher)

        teacher_count = selected_mask.sum(dim=1)
        positive_score_mask = anchor_scores > 0

        top_pool_num = min(64, total_anchor_num)
        top_score_idx = torch.topk(anchor_scores, k=top_pool_num, dim=1, largest=True, sorted=False).indices
        top_score_mask = torch.zeros_like(selected_mask)
        top_score_mask.scatter_(1, top_score_idx, True)
        top_score_mask &= positive_score_mask
        top_score_mask &= ~selected_mask

        top_score_available = top_score_mask.sum(dim=1)
        extra_teacher_needed = torch.clamp(
            torch.full_like(teacher_count, teacher_sample_num) - teacher_count,
            min=0,
        )
        extra_teacher_needed = torch.minimum(extra_teacher_needed, top_score_available)
        selected_mask |= _sample_mask_from_pool(top_score_mask, extra_teacher_needed)

        remaining_needed = total_num - selected_mask.sum(dim=1)
        if (remaining_needed < 0).any():
            raise RuntimeError("teacher-selected anchor count exceeds total candidate count")

        remaining_pool = ~selected_mask
        remaining_available = remaining_pool.sum(dim=1)
        remaining_needed = torch.minimum(remaining_needed, remaining_available)
        selected_mask |= _sample_mask_from_pool(remaining_pool, remaining_needed)

        selected_count = selected_mask.sum(dim=1)
        if not bool((selected_count == total_num).all()):
            raise RuntimeError(
                f"failed to sample enough anchors with anchor scores: got min={int(selected_count.min().item())} vs {total_num}"
            )

        random_order = torch.rand(selected_mask.shape, device=anchor_scores.device, dtype=anchor_scores.dtype)
        random_order = random_order.masked_fill(~selected_mask, 2.0)
        return torch.topk(random_order, k=total_num, dim=1, largest=False).indices.long()

    def _select_local_teacher_positive_index(
        self,
        used_index: torch.Tensor,
        teacher_indices: Optional[torch.Tensor],
        expert_distances: torch.Tensor,
    ) -> torch.Tensor:
        if teacher_indices is None:
            return torch.argmin(expert_distances, dim=-1)

        teacher_mask = (used_index.unsqueeze(-1) == teacher_indices.unsqueeze(1)).any(dim=-1)
        fallback_pos = torch.argmin(expert_distances, dim=-1)
        inf = torch.full_like(expert_distances, float('inf'))
        masked_teacher_dist = torch.where(teacher_mask, expert_distances, inf)
        teacher_pos = torch.argmin(masked_teacher_dist, dim=-1)
        has_teacher = teacher_mask.any(dim=-1)
        return torch.where(has_teacher, teacher_pos, fallback_pos)

    def _extract_planning_expert_target(self, expert_trajectory: torch.Tensor) -> torch.Tensor:
        if expert_trajectory.shape[1] >= self.num_poses + 1:
            return expert_trajectory[:, 1:self.num_poses + 1, ALL_IDX]
        if expert_trajectory.shape[1] >= self.num_poses:
            return expert_trajectory[:, :self.num_poses, ALL_IDX]
        pad_count = self.num_poses - expert_trajectory.shape[1]
        pad = expert_trajectory[:, -1:, ALL_IDX].expand(-1, pad_count, -1)
        return torch.cat([expert_trajectory[..., ALL_IDX], pad], dim=1)

    def _select_local_teacher_positive_index_from_targets(
        self,
        used_index: torch.Tensor,
        teacher_indices: Optional[torch.Tensor],
        sampled_anchor_scores: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = used_index.shape[0]
        fallback_pos = torch.zeros(batch_size, dtype=torch.long, device=used_index.device)
        if sampled_anchor_scores is not None:
            scores = sampled_anchor_scores.float()
            positive_mask = scores > 0
            score_pos = scores.argmax(dim=-1)
            fallback_pos = torch.where(positive_mask.any(dim=-1), score_pos, fallback_pos)

        if teacher_indices is None:
            return fallback_pos

        teacher_mask = (used_index.unsqueeze(-1) == teacher_indices.unsqueeze(1)).any(dim=-1)
        teacher_pos = teacher_mask.to(dtype=torch.int64).argmax(dim=-1)
        has_teacher = teacher_mask.any(dim=-1)
        return torch.where(has_teacher, teacher_pos, fallback_pos)

    def _score_candidate_anchors(
        self,
        decoder_context: Dict[str, torch.Tensor],
        anchor_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, candidate_num = anchor_indices.shape
    
        logits_chunks = []
        refined_chunks = []
        chunk_size = max(1, int(self.score_chunk_size))

        for start in range(0, candidate_num, chunk_size):
            end = min(candidate_num, start + chunk_size)
            chunk_indices = anchor_indices[:, start:end]
            chunk_num = end - start
            used_anchor = self.planner_anchor.index_select(0, chunk_indices.reshape(-1)).view(
                batch_size,
                chunk_num,
                self.num_poses,
                -1,
            )
            anchor_feat = self.anchor_encoder(
                used_anchor.reshape(batch_size * chunk_num, self.num_poses, -1)
            ).view(batch_size, chunk_num, -1)
            
            traj_query = self.ego_query_pre_branch(anchor_feat)
            decoded_query = self._run_query_decoder(
                traj_query=traj_query,
                decoder_context=decoder_context
            )
            
            refined_traj = used_anchor
            
            logits = self.planning_scorer(decoded_query).squeeze(-1)

            logits_chunks.append(logits)
            refined_chunks.append(refined_traj)

        out = {
            'logits': torch.cat(logits_chunks, dim=1),
            'refined_trajs': torch.cat(refined_chunks, dim=1),
        }

        return out

    def forward(self,
                features: FeaturesType, 
                targets: Optional[TargetsType] = None,
                ) -> Dict[str, torch.Tensor]:
        
        scene_feature: SceneFeature = features['scene_feature']
        is_train = self.training
        device = features['scene_feature'].ego_feature.ego_current_state.device
        batch_size = scene_feature.ego_feature.ego_current_state.shape[0]

        expert_target_feature = targets.get('expert_trajectory') if targets is not None else None
        expert_trajectory = expert_target_feature.data if expert_target_feature is not None else None
        agent_predictions: Optional[AgentPrediction] = targets.get('agent_prediction', None) if targets is not None else None
        anchor_indices = targets.get('anchor_indice', None) if targets is not None else None
        anchor_scores = targets.get('anchor_scores', None) if targets is not None else None
        
        total_anchor_num = self.planner_anchor.shape[0]
        out: Dict[str, torch.Tensor] = {}
        decoder_context = self._encode_scene_context(scene_feature)
        decoder_context = self._run_agent_context_decoder(decoder_context)
        decoder_context, agent_pred_trajs, agent_pred_confidence, top1_agent_pred, agent_future_mask = self._run_prediction_modules(
            scene_feature,
            decoder_context,
        )
        if self.use_prediction and top1_agent_pred is not None and agent_future_mask is not None:
            out['agent_prediction'] = AgentPrediction(
                agent_future_state=top1_agent_pred,
                agent_future_mask=agent_future_mask,
            )
            out['agent_prediction_confidence'] = agent_pred_confidence

        candidate_num = min(
            self.train_anchor_num if is_train else self.test_anchor_num,
            total_anchor_num,
        )

        # do not apply dynamic score-based sampling when training
        dynamic_scores = torch.ones(batch_size, total_anchor_num, device=device)

        teacher_indices = None
        target_teacher_indices = None
        anchor_score_targets = None
        use_anchor_score_guidance = False
        use_full_anchor_teacher_sampling = False
        expert_target = None
        
        if expert_trajectory is not None:
            expert_target = expert_trajectory[:, :self.num_poses, ALL_IDX]
            
            target_teacher_indices = self._resolve_teacher_indices_from_targets(
                anchor_indices=anchor_indices,
                batch_size=batch_size,
                device=device,
            )
            anchor_score_targets = self._resolve_anchor_scores_from_targets(
                anchor_scores=anchor_scores,
                batch_size=batch_size,
                device=device,
            )
            use_anchor_score_guidance = (
                target_teacher_indices is not None
                and anchor_score_targets is not None
                and anchor_score_targets.shape[1] == total_anchor_num
            )
            use_full_anchor_teacher_sampling = (
                use_anchor_score_guidance
                and self.teacher_anchor_ratio >= 1.0 - 1e-6
            )
            if use_anchor_score_guidance:
                teacher_indices = target_teacher_indices
            else:
                teacher_indices = (
                    self._compose_teacher_indices(
                        target_teacher_indices=target_teacher_indices,
                        expert_anchor_target=expert_target,
                        candidate_num=candidate_num,
                    )
                    if self.use_expert_distance_guidance
                    else target_teacher_indices
                )

        if use_full_anchor_teacher_sampling:
            used_index = self._sample_candidate_indices(
                dynamic_scores=torch.ones_like(dynamic_scores),
                total_num=candidate_num,
                teacher_indices=teacher_indices,
            )
            sampled_anchor_scores = None
        elif use_anchor_score_guidance:
            used_index = self._sample_candidate_indices_with_anchor_scores(
                anchor_scores=anchor_score_targets,
                total_num=candidate_num,
                teacher_indices=teacher_indices,
            )
            sampled_anchor_scores = anchor_score_targets.gather(1, used_index)
        else:
            used_index = self._sample_candidate_indices(
                dynamic_scores=dynamic_scores,
                total_num=candidate_num,
                teacher_indices=teacher_indices,
            )
            sampled_anchor_scores = None

        score_out = self._score_candidate_anchors(
            decoder_context=decoder_context,
            anchor_indices=used_index,
        )

        planner_logits = score_out['logits']
        candidate_trajs = score_out['refined_trajs']
        planner_logits = self._debug_tensor(planner_logits, "planner_logits", clamp=1e3)
        candidate_trajs = self._debug_tensor(candidate_trajs, "candidate_trajs", clamp=1e6)
        best_local_idx = torch.argmax(planner_logits, dim=-1)
        batch_index = torch.arange(batch_size, device=planner_logits.device)
        planned_traj = candidate_trajs[batch_index, best_local_idx]
        planned_traj = self._debug_tensor(planned_traj, "planned_traj", clamp=1e6)

        out['trajectory'] = planned_traj
        out['candidate_trajectories'] = candidate_trajs
        out['candidate_scores'] = planner_logits
        out['indices'] = used_index
        out['mean_expert_distance'] = planned_traj.new_tensor(0.0)
        

        if expert_trajectory is not None:
            if self.use_expert_distance_guidance:
                expert_distances = score_anchors_by_expert_traj(
                    expert_target.float(),
                    candidate_trajs.float(),
                ).float()
                teacher_pos_idx = self._select_local_teacher_positive_index(
                    used_index=used_index,
                    teacher_indices=teacher_indices,
                    expert_distances=expert_distances,
                )
            else:
                expert_distances = torch.zeros_like(planner_logits, dtype=torch.float32)
                teacher_pos_idx = self._select_local_teacher_positive_index_from_targets(
                    used_index=used_index,
                    teacher_indices=teacher_indices,
                    sampled_anchor_scores=sampled_anchor_scores,
                )
            
            loss_dict = self._compute_loss(
                planned_traj=planned_traj,
                expert_traj=expert_target,
                agent_predictions=agent_predictions,
                agent_pred_trajs=agent_pred_trajs if self.use_prediction else None,
                agent_pred_confidence=agent_pred_confidence if self.use_prediction else None,
                logits=planner_logits.float(),
                expert_dist=expert_distances.float(),
                teacher_pos_idx=teacher_pos_idx,
                sampled_anchor_scores=sampled_anchor_scores,
            )
            out['mean_expert_distance'] = loss_dict.get('mean_expert_distance', planned_traj.new_tensor(0.0))
            out['loss_dict'] = loss_dict

        return out
    
    
    @torch.no_grad()
    def sample_trajectories(self,
                            features: FeaturesType,
                            num_samples: int = 5,
                            temperature: float = 1.0,
                            top_k: int = 0,
                            top_p: float = 0.0,
                            deterministic_first: bool = True,
                            filter_dynamic: bool = False,
                            max_chunk_num: int = 1) -> Dict[str, torch.Tensor]:
        assert temperature > 0, "temperature must be > 0"
        self.eval()
        scene_feature: SceneFeature = features['scene_feature']
        batch_size = scene_feature.ego_feature.ego_current_state.shape[0]
        out: Dict[str, torch.Tensor] = {}
        decoder_context = self._encode_scene_context(scene_feature)
        decoder_context = self._run_agent_context_decoder(decoder_context)
        decoder_context, agent_pred_trajs, agent_pred_confidence, top1_agent_pred, agent_future_mask = self._run_prediction_modules(
            scene_feature,
            decoder_context,
        )
        if self.use_prediction and top1_agent_pred is not None and agent_future_mask is not None:
            out['agent_prediction'] = AgentPrediction(
                agent_future_state=top1_agent_pred,
                agent_future_mask=agent_future_mask,
            )
            out['agent_prediction_confidence'] = agent_pred_confidence
        total_anchor_num = self.planner_anchor.shape[0]
        used_index = torch.arange(
            total_anchor_num,
            device=self.planner_anchor.device,
        ).unsqueeze(0).expand(batch_size, -1)
        score_out = self._score_candidate_anchors(
            decoder_context=decoder_context,
            anchor_indices=used_index,
        )
        all_logp = F.log_softmax(score_out['logits'], dim=-1)
        num_samples = int(max(1, min(int(num_samples), total_anchor_num)))
        scores, picks = torch.topk(all_logp, k=num_samples, dim=-1, largest=True, sorted=True)

        trajectories = score_out['refined_trajs'].gather(
            1,
            picks.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_poses, len(ALL_IDX)),
        )
        indices = used_index.gather(1, picks).unsqueeze(-1)
        out.update({
            'trajectories': trajectories,
            'indices': indices,
            'scores': scores,
            'all_logp': all_logp,
            'all_indices': used_index,
            'candidate_scores': score_out['logits'],
        })
        if 'moe_router_logits' in score_out:
            out['moe_router_logits'] = score_out['moe_router_logits']
            out['moe_router_weights'] = score_out['moe_router_weights']
            out['moe_topk_indices'] = score_out['moe_topk_indices']
            out['moe_topk_weights'] = score_out['moe_topk_weights']
        return out
    
    def score_trajectories(
        self,
        scene_features: SceneFeature,
        indices: torch.Tensor,
        no_grad: bool = True,
        return_step_logp: bool = True,
        return_entropy: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if indices.dim() == 1:
            indices = indices.unsqueeze(-1)
        if indices.dim() != 2 or indices.shape[1] != 1:
            raise ValueError(f"indices must be [B, 1], got shape={tuple(indices.shape)}")

        with torch.set_grad_enabled(not no_grad):
            scene_feature: SceneFeature = scene_features
            batch_size = scene_feature.ego_feature.ego_current_state.shape[0]
            total_anchor_num = self.planner_anchor.shape[0]
            if indices.shape[0] != batch_size:
                raise ValueError(f"indices batch mismatch: indices.shape[0]={indices.shape[0]} vs B={batch_size}")
            decoder_context = self._encode_scene_context(scene_feature)
            decoder_context = self._run_agent_context_decoder(decoder_context)
            decoder_context, agent_pred_trajs, agent_pred_confidence, top1_agent_pred, agent_future_mask = self._run_prediction_modules(
                scene_feature,
                decoder_context,
            )

            full_index = torch.arange(total_anchor_num, device=self.planner_anchor.device).unsqueeze(0).expand(batch_size, -1)
            score_out = self._score_candidate_anchors(
                decoder_context=decoder_context,
                anchor_indices=full_index,
            )
            logits = score_out['logits']
            logp_all = F.log_softmax(logits, dim=-1)
            target_index = indices[:, 0].to(device=logp_all.device, dtype=torch.long)
            target_logp = logp_all.gather(1, target_index.unsqueeze(1)).squeeze(1)

            out: Dict[str, torch.Tensor] = {
                "target_logp": target_logp,
                "all_logp": logp_all,
            }
            if 'moe_router_logits' in score_out:
                out["moe_router_logits"] = score_out['moe_router_logits']
                out["moe_router_weights"] = score_out['moe_router_weights']
                out["moe_topk_indices"] = score_out['moe_topk_indices']
                out["moe_topk_weights"] = score_out['moe_topk_weights']
            if self.use_prediction:
                out["agent_prediction_modes"] = agent_pred_trajs
                out["agent_prediction"] = AgentPrediction(
                    agent_future_state=top1_agent_pred,
                    agent_future_mask=agent_future_mask,
                )
                out["agent_prediction_confidence"] = agent_pred_confidence
            if return_step_logp:
                out["step_logp"] = logp_all.unsqueeze(1)
            if return_entropy:
                prob_all = logp_all.exp()
                entropy = -(prob_all * logp_all).sum(dim=-1)
                out["step_entropy"] = entropy.unsqueeze(1)
                out["entropy_sum"] = entropy
                out["entropy_mean"] = entropy
            return out

    def score_candidate_subset(
        self,
        scene_features: SceneFeature,
        anchor_indices: torch.Tensor,
        no_grad: bool = True,
        return_prediction: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if anchor_indices.dim() == 1:
            anchor_indices = anchor_indices.unsqueeze(0)
        if anchor_indices.dim() != 2:
            raise ValueError(f"anchor_indices must be [B, K], got shape={tuple(anchor_indices.shape)}")

        with torch.set_grad_enabled(not no_grad):
            scene_feature: SceneFeature = scene_features
            batch_size = scene_feature.ego_feature.ego_current_state.shape[0]
            if anchor_indices.shape[0] != batch_size:
                raise ValueError(
                    f"anchor_indices batch mismatch: anchor_indices.shape[0]={anchor_indices.shape[0]} vs B={batch_size}"
                )

            decoder_context = self._encode_scene_context(scene_feature)
            decoder_context = self._run_agent_context_decoder(decoder_context)
            decoder_context, agent_pred_trajs, agent_pred_confidence, top1_agent_pred, agent_future_mask = self._run_prediction_modules(
                scene_feature,
                decoder_context,
            )

            score_out = self._score_candidate_anchors(
                decoder_context=decoder_context,
                anchor_indices=anchor_indices.long(),
            )
            
            logits = score_out['logits']
            logp = F.log_softmax(logits, dim=-1)
            out: Dict[str, torch.Tensor] = {
                "candidate_scores": logits,
                "candidate_log_probs": logp,
                "indices": anchor_indices.long(),
            }
            if 'moe_router_logits' in score_out:
                out["moe_router_logits"] = score_out['moe_router_logits']
                out["moe_router_weights"] = score_out['moe_router_weights']
                out["moe_topk_indices"] = score_out['moe_topk_indices']
                out["moe_topk_weights"] = score_out['moe_topk_weights']
            if return_prediction and self.use_prediction:
                out["agent_prediction_modes"] = agent_pred_trajs
                out["agent_prediction_confidence"] = agent_pred_confidence
                out["agent_prediction"] = AgentPrediction(
                    agent_future_state=top1_agent_pred,
                    agent_future_mask=agent_future_mask,
                )
            return out

    
    
    def _compute_prediction_loss_package(
        self,
        planned_traj: torch.Tensor,
        agent_predictions: Optional[AgentPrediction],
        agent_pred_trajs: Optional[torch.Tensor],
        agent_pred_confidence: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        batch_size = planned_traj.shape[0]
        zeros_batch = planned_traj.new_zeros(batch_size)
        zero_scalar = planned_traj.new_tensor(0.0)
        out = {
            'prediction_loss_batch': zeros_batch,
            'prediction_regression_loss_batch': zeros_batch,
            'prediction_confidence_loss_batch': zeros_batch,
            'prediction_diversity_loss_batch': zeros_batch,
            'prediction_ade_batch': zeros_batch,
            'prediction_fde_batch': zeros_batch,
            'prediction_loss': zero_scalar,
            'prediction_regression_loss': zero_scalar,
            'prediction_confidence_loss': zero_scalar,
            'prediction_diversity_loss': zero_scalar,
            'prediction_ade': zero_scalar,
            'prediction_fde': zero_scalar,
        }

        if (not self.use_prediction) or (agent_predictions is None) or (agent_pred_trajs is None):
            return out

        agent_traj_gt = torch.nan_to_num(
            agent_predictions.agent_future_state.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        agent_traj_mask = torch.nan_to_num(
            agent_predictions.agent_future_mask.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        pred = torch.nan_to_num(
            agent_pred_trajs.float(),
            nan=0.0,
            posinf=1e3,
            neginf=-1e3,
        )
        pred_conf = (
            torch.nan_to_num(
                agent_pred_confidence.float(),
                nan=0.0,
                posinf=30.0,
                neginf=-30.0,
            )
            if agent_pred_confidence is not None
            else None
        )

        num_agents = min(pred.shape[1], agent_traj_gt.shape[1], agent_traj_mask.shape[1])
        if num_agents <= 0:
            return out

        pred = pred[:, :num_agents, ...]
        agent_traj_gt = agent_traj_gt[:, :num_agents, ...]
        agent_traj_mask = agent_traj_mask[:, :num_agents, ...]
        if pred_conf is not None:
            pred_conf = pred_conf[:, :num_agents, ...]

        gt_pred = agent_traj_gt[..., :self.prediction_steps, :]
        gt_mask = agent_traj_mask[..., :self.prediction_steps]

        t_idx = torch.arange(
            self.prediction_steps,
            device=pred.device,
            dtype=pred.dtype,
        )
        time_weight = 1.0 / torch.pow(1.0 + t_idx, self.prediction_time_weight_gamma)
        time_weight = time_weight * (self.prediction_steps / time_weight.sum().clamp_min(1e-6))
        time_weight = time_weight.view(1, 1, 1, self.prediction_steps)

        weighted_mask = gt_mask.unsqueeze(2) * time_weight
        denom = weighted_mask.sum(dim=-1).clamp_min(1.0)
        valid_agent = gt_mask.sum(dim=-1) > 0
        gt_modes = gt_pred.unsqueeze(2)

        xy_dist = torch.linalg.norm(pred[..., :2] - gt_modes[..., :2], dim=-1)
        ade_xy = (xy_dist * weighted_mask).sum(dim=-1) / denom

        yaw_delta = torch.atan2(
            torch.sin(pred[..., 2] - gt_modes[..., 2]),
            torch.cos(pred[..., 2] - gt_modes[..., 2]),
        )
        yaw_mean = (yaw_delta.abs() * weighted_mask).sum(dim=-1) / denom

        vel_abs = (pred[..., 3] - gt_modes[..., 3]).abs()
        vel_mean = (vel_abs * weighted_mask).sum(dim=-1) / denom

        last_idx = gt_mask.long().sum(dim=-1).sub(1).clamp_min(0)
        gather_idx_pred = last_idx.unsqueeze(2).unsqueeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            pred.shape[2],
            1,
            pred.shape[-1],
        )
        pred_last = pred.gather(3, gather_idx_pred).squeeze(3)

        gather_idx_gt = last_idx.unsqueeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            1,
            gt_pred.shape[-1],
        )
        gt_last = gt_pred.gather(2, gather_idx_gt).squeeze(2)

        fde_xy = torch.linalg.norm(pred_last[..., :2] - gt_last.unsqueeze(2)[..., :2], dim=-1)
        yaw_end = torch.atan2(
            torch.sin(pred_last[..., 2] - gt_last.unsqueeze(2)[..., 2]),
            torch.cos(pred_last[..., 2] - gt_last.unsqueeze(2)[..., 2]),
        ).abs()
        vel_end = (pred_last[..., 3] - gt_last.unsqueeze(2)[..., 3]).abs()

        mode_cost = (
            self.prediction_ade_weight * ade_xy
            + self.prediction_fde_weight * fde_xy
            + self.prediction_yaw_weight * yaw_mean
            + self.prediction_yaw_end_weight * yaw_end
            + self.prediction_vel_weight * vel_mean
            + self.prediction_vel_end_weight * vel_end
        )

        best_mode_idx = mode_cost.argmin(dim=-1)
        best_mode_cost = mode_cost.gather(2, best_mode_idx.unsqueeze(-1)).squeeze(-1)
        best_mode_ade = ade_xy.gather(2, best_mode_idx.unsqueeze(-1)).squeeze(-1)
        best_mode_fde = fde_xy.gather(2, best_mode_idx.unsqueeze(-1)).squeeze(-1)

        valid_agent_float = valid_agent.float()
        valid_agent_count = valid_agent_float.sum(dim=-1).clamp_min(1.0)
        prediction_regression_loss_batch = (best_mode_cost * valid_agent_float).sum(dim=-1) / valid_agent_count
        prediction_ade_batch = (best_mode_ade * valid_agent_float).sum(dim=-1) / valid_agent_count
        prediction_fde_batch = (best_mode_fde * valid_agent_float).sum(dim=-1) / valid_agent_count
        batch_has_valid = valid_agent.any(dim=-1)
        prediction_regression_loss = (
            prediction_regression_loss_batch[batch_has_valid].mean()
            if batch_has_valid.any()
            else zero_scalar
        )
        prediction_ade = (
            prediction_ade_batch[batch_has_valid].mean()
            if batch_has_valid.any()
            else zero_scalar
        )
        prediction_fde = (
            prediction_fde_batch[batch_has_valid].mean()
            if batch_has_valid.any()
            else zero_scalar
        )

        prediction_confidence_loss_batch = planned_traj.new_zeros(batch_size)
        prediction_confidence_loss = zero_scalar
        if pred_conf is not None and valid_agent.any():
            confidence_loss_values = F.cross_entropy(
                pred_conf[valid_agent],
                best_mode_idx[valid_agent],
                reduction='none',
            )
            valid_agent_indices = valid_agent.nonzero(as_tuple=False)[:, 0]
            prediction_confidence_loss_batch.scatter_add_(0, valid_agent_indices, confidence_loss_values)
            prediction_confidence_loss_batch = prediction_confidence_loss_batch / valid_agent_count
            prediction_confidence_loss = (
                prediction_confidence_loss_batch[batch_has_valid].mean()
                if batch_has_valid.any()
                else zero_scalar
            )

        prediction_diversity_loss_batch = planned_traj.new_zeros(batch_size)
        prediction_diversity_loss = zero_scalar
        if self.prediction_diversity_weight > 0.0 and pred.shape[2] > 1 and valid_agent.any():
            mode_i, mode_j = torch.triu_indices(pred.shape[2], pred.shape[2], offset=1, device=pred.device)
            endpoint_xy = pred[..., -1, :2]
            pairwise_delta = endpoint_xy[:, :, mode_i, :] - endpoint_xy[:, :, mode_j, :]
            pairwise_dist = torch.linalg.norm(pairwise_delta, dim=-1)
            diversity_penalty = F.relu(self.prediction_diversity_margin - pairwise_dist)
            diversity_loss_agent = diversity_penalty.mean(dim=-1)
            prediction_diversity_loss_batch = (diversity_loss_agent * valid_agent_float).sum(dim=-1) / valid_agent_count
            prediction_diversity_loss = (
                prediction_diversity_loss_batch[batch_has_valid].mean()
                if batch_has_valid.any()
                else zero_scalar
            )

        prediction_loss_batch = (
            prediction_regression_loss_batch
            + self.prediction_confidence_loss_weight * prediction_confidence_loss_batch
            + self.prediction_diversity_weight * prediction_diversity_loss_batch
        )
        prediction_loss = prediction_loss_batch.mean()

        out.update({
            'prediction_loss_batch': prediction_loss_batch,
            'prediction_regression_loss_batch': prediction_regression_loss_batch,
            'prediction_confidence_loss_batch': prediction_confidence_loss_batch,
            'prediction_diversity_loss_batch': prediction_diversity_loss_batch,
            'prediction_ade_batch': prediction_ade_batch,
            'prediction_fde_batch': prediction_fde_batch,
            'prediction_loss': prediction_loss,
            'prediction_regression_loss': prediction_regression_loss,
            'prediction_confidence_loss': prediction_confidence_loss,
            'prediction_diversity_loss': prediction_diversity_loss,
            'prediction_ade': prediction_ade,
            'prediction_fde': prediction_fde,
        })
        return out


    def _compute_loss(
        self,
        planned_traj: torch.Tensor,
        expert_traj: torch.Tensor,
        agent_predictions: Optional[AgentPrediction],
        agent_pred_trajs: Optional[torch.Tensor],
        agent_pred_confidence: Optional[torch.Tensor],
        logits: torch.Tensor,
        expert_dist: torch.Tensor,
        teacher_pos_idx: torch.Tensor,
        regression_source_traj: Optional[torch.Tensor] = None,
        sampled_anchor_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        loss_dict: Dict[str, torch.Tensor] = {}
        batch_size = planned_traj.shape[0]
        T_use = min(planned_traj.shape[1], expert_traj.shape[1])
        regression_source = planned_traj if regression_source_traj is None else regression_source_traj
        planned_xy = regression_source[:, :T_use, :2]
        expert_xy = expert_traj[:, :T_use, :2]
        regression_xy_loss_batch = F.smooth_l1_loss(planned_xy, expert_xy, reduction="none").mean(dim=(-1, -2))
        regression_yaw_loss_batch = planned_traj.new_zeros(batch_size)
        if self.regression_yaw_loss_weight > 0.0 and regression_source.shape[-1] > 2 and expert_traj.shape[-1] > 2:
            planned_yaw = regression_source[:, :T_use, 2]
            expert_yaw = expert_traj[:, :T_use, 2]
            yaw_delta = torch.atan2(
                torch.sin(planned_yaw - expert_yaw),
                torch.cos(planned_yaw - expert_yaw),
            )
            regression_yaw_loss_batch = F.smooth_l1_loss(
                yaw_delta,
                torch.zeros_like(yaw_delta),
                reduction="none",
            ).mean(dim=-1)
        regression_loss_batch = regression_xy_loss_batch + self.regression_yaw_loss_weight * regression_yaw_loss_batch
        regression_loss = regression_loss_batch.mean()
        

        pos_idx = teacher_pos_idx
        teacher_ce_loss_batch = planned_traj.new_zeros(batch_size)
        anchor_score_kl_loss_batch = planned_traj.new_zeros(batch_size)
        anchor_score_rank_loss_batch = planned_traj.new_zeros(batch_size)
        anchor_score_neg_loss_batch = planned_traj.new_zeros(batch_size)
        teacher_ce_loss = planned_traj.new_tensor(0.0)
        anchor_score_kl_loss = planned_traj.new_tensor(0.0)
        anchor_score_rank_loss = planned_traj.new_tensor(0.0)
        anchor_score_neg_loss = planned_traj.new_tensor(0.0)
        use_anchor_score_guidance = sampled_anchor_scores is not None

        if use_anchor_score_guidance:
            logits = logits.float()
            expert_dist = expert_dist.float()
            sampled_anchor_scores = sampled_anchor_scores.to(dtype=logits.dtype).clamp_min(0.0)
            positive_mask = sampled_anchor_scores > 0
            has_positive = positive_mask.any(dim=-1)
            if has_positive.any():
                teacher_target_prob = self._build_teacher_soft_target(
                    pos_idx=pos_idx[has_positive],
                    positive_mask=positive_mask[has_positive],
                    dtype=logits.dtype,
                )
                teacher_ce_loss_batch[has_positive] = -(
                    teacher_target_prob * F.log_softmax(logits[has_positive], dim=-1)
                ).sum(dim=-1)
                teacher_ce_loss = teacher_ce_loss_batch[has_positive].mean()

                logits_valid = logits[has_positive]
                pos_idx_valid = pos_idx[has_positive]
                expert_dist_valid = expert_dist[has_positive]
                positive_mask_valid = positive_mask[has_positive]
                target_scores = sampled_anchor_scores[has_positive]
                if self.use_anchor_score_kl_loss:
                    target_prob = target_scores / target_scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    log_prob = F.log_softmax(logits_valid, dim=-1)
                    soft_loss = -(target_prob * log_prob).sum(dim=-1)
                    anchor_score_kl_loss_batch[has_positive] = soft_loss
                    anchor_score_kl_loss = soft_loss.mean()

                teacher_dist = expert_dist_valid.gather(1, pos_idx_valid.unsqueeze(1)).squeeze(1)
                delta_dist = (expert_dist_valid - teacher_dist.unsqueeze(1)).clamp_min(0.0)
                delta_scale = delta_dist.mean(dim=-1, keepdim=True).clamp_min(self.neg_delta_floor)
                normalized_delta = delta_dist / delta_scale

                if self.anchor_score_neg_loss_weight > 0.0:
                    prob_valid = F.softmax(logits_valid, dim=-1)
                    prob_eps = max(float(torch.finfo(prob_valid.dtype).eps) * 10.0, 1e-6)
                    prob_valid = prob_valid.clamp(min=prob_eps, max=1.0 - prob_eps)
                    negative_weight = ((normalized_delta - self.neg_delta_margin) / self.neg_delta_slope).clamp(min=0.0, max=100.0)
                    negative_mask = ~positive_mask_valid
                    negative_mask.scatter_(1, pos_idx_valid.unsqueeze(1), False)
                    negative_weight = torch.where(negative_mask, negative_weight, torch.zeros_like(negative_weight))
                    neg_term = negative_mask.float() * (prob_valid ** 2.0) * torch.log1p(-prob_valid)
                    neg_loss_values = -(neg_term * negative_weight).sum(dim=-1) / negative_weight.sum(dim=-1).clamp_min(1.0)
                    valid_batch_indices = has_positive.nonzero(as_tuple=False).squeeze(1)
                    anchor_score_neg_loss_batch[valid_batch_indices] = neg_loss_values
                    anchor_score_neg_loss = neg_loss_values.mean()

                negative_mask = ~positive_mask_valid
                negative_mask.scatter_(1, pos_idx_valid.unsqueeze(1), False)
                hard_negative_mask = negative_mask & (normalized_delta >= 1.0)
                hard_negative_exists = hard_negative_mask.any(dim=-1, keepdim=True)
                effective_negative_mask = torch.where(hard_negative_exists, hard_negative_mask, negative_mask)
                valid_rank = effective_negative_mask.any(dim=-1)
                if valid_rank.any():
                    neg_priority = logits_valid + normalized_delta.detach()
                    neg_priority = neg_priority.masked_fill(~effective_negative_mask, float('-inf'))
                    hard_neg_idx = neg_priority.argmax(dim=-1)
                    pos_logit = logits_valid.gather(1, pos_idx_valid.unsqueeze(1)).squeeze(1)
                    neg_logit = logits_valid.gather(1, hard_neg_idx.unsqueeze(1)).squeeze(1)
                    rank_gap = pos_logit - neg_logit
                    rank_loss_values = F.relu(self.neg_delta_margin - rank_gap)[valid_rank]
                    valid_batch_indices = has_positive.nonzero(as_tuple=False).squeeze(1)[valid_rank]
                    anchor_score_rank_loss_batch[valid_batch_indices] = rank_loss_values
                    anchor_score_rank_loss = rank_loss_values.mean()

            anchor_score_rank_loss = anchor_score_rank_loss_batch.mean()
            dist_loss_batch = (
                self.teacher_ce_weight * teacher_ce_loss_batch
                + self.anchor_score_kl_weight * anchor_score_kl_loss_batch
                + anchor_score_rank_loss_batch
                + self.anchor_score_neg_loss_weight * anchor_score_neg_loss_batch
            )
            dist_loss = dist_loss_batch.mean()
        else:
            target_cost = expert_dist
            target_cost = self._debug_tensor(target_cost, "planning_loss.target_cost", clamp=1e6)
            target_prob = F.softmax(
                -target_cost / max(float(self.imitation_target_tau), 1e-6),
                dim=-1,
            ).clamp_min(1e-8)
            target_prob = self._debug_tensor(target_prob, "planning_loss.target_prob", clamp=1.0)
            log_prob = F.log_softmax(logits, dim=-1)
            log_prob = self._debug_tensor(log_prob, "planning_loss.log_prob", clamp=1e6)
            dist_loss_batch = F.kl_div(log_prob, target_prob, reduction='none').sum(dim=-1)
            dist_loss = dist_loss_batch.mean()

        dist_loss = self._debug_tensor(dist_loss, "planning_loss.dist_loss", clamp=1e6)
        dist_loss_batch = self._debug_tensor(dist_loss_batch, "planning_loss.dist_loss_batch", clamp=1e6)

        pos_dist = expert_dist.gather(1, pos_idx.unsqueeze(1)).squeeze(1)
        mean_expert_dist = pos_dist.mean()

        prediction_pkg = self._compute_prediction_loss_package(
            planned_traj=planned_traj,
            agent_predictions=agent_predictions,
            agent_pred_trajs=agent_pred_trajs,
            agent_pred_confidence=agent_pred_confidence,
        )
        prediction_loss_batch = prediction_pkg['prediction_loss_batch']
        prediction_reg_loss_batch = prediction_pkg['prediction_regression_loss_batch']
        prediction_confidence_loss_batch = prediction_pkg['prediction_confidence_loss_batch']
        prediction_diversity_loss_batch = prediction_pkg['prediction_diversity_loss_batch']
        prediction_ade_batch = prediction_pkg['prediction_ade_batch']
        prediction_fde_batch = prediction_pkg['prediction_fde_batch']
        prediction_loss = prediction_pkg['prediction_loss']
        prediction_reg_loss = prediction_pkg['prediction_regression_loss']
        prediction_confidence_loss = prediction_pkg['prediction_confidence_loss']
        prediction_diversity_loss = prediction_pkg['prediction_diversity_loss']


        total_loss_batch = (
            self.regression_loss_weight * regression_loss_batch
            + self.classification_loss_weight * dist_loss_batch
            + self.prediction_loss_weight * prediction_loss_batch
        )
        total_loss = total_loss_batch.mean()

        pred_top1_idx = torch.argmax(logits, dim=-1)
        top1_hit = (pred_top1_idx == teacher_pos_idx).float().mean()

        prob = F.softmax(logits, dim=-1)
        teacher_prob = prob.gather(1, teacher_pos_idx.unsqueeze(1)).squeeze(1).mean()

        loss_dict["top1_hit_rate"] = top1_hit
        loss_dict["teacher_prob"] = teacher_prob

        loss_dict['dist_loss'] = dist_loss_batch
        loss_dict['classification_loss'] = dist_loss_batch
        if use_anchor_score_guidance:
            loss_dict['teacher_ce_loss'] = teacher_ce_loss_batch
            loss_dict['anchor_score_kl_loss'] = anchor_score_kl_loss_batch
            loss_dict['anchor_score_rank_loss'] = anchor_score_rank_loss_batch
            loss_dict['anchor_score_neg_loss'] = anchor_score_neg_loss_batch
            loss_dict['weighted_teacher_ce_loss'] = self.teacher_ce_weight * teacher_ce_loss_batch
            loss_dict['weighted_anchor_score_kl_loss'] = self.anchor_score_kl_weight * anchor_score_kl_loss_batch
            loss_dict['weighted_anchor_score_neg_loss'] = self.anchor_score_neg_loss_weight * anchor_score_neg_loss_batch
        loss_dict['regression_loss'] = regression_loss_batch
        loss_dict['regression_xy_loss'] = regression_xy_loss_batch
        if self.regression_yaw_loss_weight > 0.0:
            loss_dict['regression_yaw_loss'] = regression_yaw_loss_batch
        if self.use_prediction and (agent_predictions is not None):
            loss_dict['prediction_loss'] = prediction_loss_batch
            loss_dict['prediction_regression_loss'] = prediction_reg_loss_batch
            loss_dict['prediction_confidence_loss'] = prediction_confidence_loss_batch
            loss_dict['prediction_diversity_loss'] = prediction_diversity_loss_batch
            loss_dict['prediction_ade'] = prediction_ade_batch
            loss_dict['prediction_fde'] = prediction_fde_batch
        
        loss_dict['mean_expert_distance'] = mean_expert_dist
        loss_dict['mean_expert_distance_batch'] = pos_dist
        loss_dict['total_loss'] = total_loss_batch
        return loss_dict


    def _build_teacher_soft_target(
        self,
        pos_idx: torch.Tensor,
        positive_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, candidate_num = positive_mask.shape
        target = torch.zeros(batch_size, candidate_num, device=positive_mask.device, dtype=dtype)
        target.scatter_(1, pos_idx.unsqueeze(1), 1.0)

        smoothing = float(self.teacher_ce_label_smoothing)
        if smoothing <= 0.0 or candidate_num <= 1:
            return target

        support_mask = positive_mask.clone()
        support_mask.scatter_(1, pos_idx.unsqueeze(1), False)
        support_count = support_mask.sum(dim=-1, keepdim=True)

        fallback_mask = torch.ones_like(positive_mask)
        fallback_mask.scatter_(1, pos_idx.unsqueeze(1), False)
        fallback_count = fallback_mask.sum(dim=-1, keepdim=True)

        use_positive_support = support_count > 0
        support_prob = torch.where(
            use_positive_support,
            support_mask.float() / support_count.clamp_min(1).float(),
            fallback_mask.float() / fallback_count.clamp_min(1).float(),
        ).to(dtype=dtype)

        return (1.0 - smoothing) * target + smoothing * support_prob

    def freeze_encoder(self):
        """
        Freeze the parameters of the state encoder and prediction head.
        """
        for param in self.state_encoder.parameters():
            param.requires_grad = False
        if hasattr(self, 'prediction_head'):
            for param in self.prediction_head.parameters():
                param.requires_grad = False
        if hasattr(self, 'agent_decoder'):
            for param in self.agent_decoder.parameters():
                param.requires_grad = False
        if hasattr(self, 'type_embed'):
            for param in self.type_embed.parameters():
                param.requires_grad = False
        for param in self.anchor_encoder.parameters():
            param.requires_grad = False


class TemporalMLP(nn.Module):
    def __init__(self, num_steps: int, num_layers: int = 2, hidden_scale: int = 2) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                LayerNorm(num_steps),
                nn.Linear(num_steps, hidden_scale * num_steps),
                nn.GELU(),
                nn.Linear(hidden_scale * num_steps, num_steps),
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(-1, -2)
        for layer in self.layers:
            x_t = x_t + layer(x_t)
        return x_t.transpose(-1, -2)


class PredictionHead(nn.Module):
    def __init__(
        self,
        dim,
        future_steps,
        num_modes: int = 4,
        dt: float = 0.5,
        use_cv_delta_prediction: bool = False,
        delta_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.future_steps = future_steps
        self.num_modes = num_modes
        self.dt = float(dt)
        self.use_cv_delta_prediction = bool(use_cv_delta_prediction)
        self.delta_scale = float(delta_scale)
        self.input_proj = Mlp(dim, 2 * dim, dim)
        self.mode_embed = nn.Embedding(num_modes, dim)
        self.time_embed = nn.Parameter(torch.randn(1, 1, 1, future_steps, dim) * 0.02)
        self.temporal_decoder = TemporalMLP(future_steps, num_layers=2, hidden_scale=2)
        self.channel_decoder = Mlp(dim, 2 * dim, dim)

        self.pos_predictor = nn.Linear(dim, 2)
        self.yaw_predictor = nn.Linear(dim, 2)
        self.vel_predictor = nn.Linear(dim, 1)
        self.confidence_predictor = Mlp(dim, 2 * dim, 1)

    def _build_cv_base_trajectory(
        self,
        agent_current_state: torch.Tensor,
        target_agents: int,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, src_agents, state_dim = agent_current_state.shape
        aligned_agents = min(src_agents, target_agents)
        current = agent_current_state.new_zeros(batch_size, target_agents, state_dim)
        current[:, :aligned_agents] = agent_current_state[:, :aligned_agents]

        # Add an explicit time axis so all base states broadcast against [1, 1, T, 1].
        x0 = current[..., 0:1].unsqueeze(2)
        y0 = current[..., 1:2].unsqueeze(2)
        yaw0 = current[..., 2:3].unsqueeze(2)
        v0 = current[..., 3:4].unsqueeze(2)

        t = torch.arange(
            1,
            self.future_steps + 1,
            device=agent_current_state.device,
            dtype=target_dtype,
        ).view(1, 1, self.future_steps, 1) * self.dt

        cos_yaw = torch.cos(yaw0).to(dtype=target_dtype)
        sin_yaw = torch.sin(yaw0).to(dtype=target_dtype)
        x = x0.to(dtype=target_dtype) + v0.to(dtype=target_dtype) * cos_yaw * t
        y = y0.to(dtype=target_dtype) + v0.to(dtype=target_dtype) * sin_yaw * t
        yaw = yaw0.to(dtype=target_dtype).expand(-1, -1, self.future_steps, -1)
        vel = v0.to(dtype=target_dtype).expand(-1, -1, self.future_steps, -1)
        return torch.cat([x, y, yaw, vel], dim=-1)

    def forward(
        self,
        x,
        agent_current_state: Optional[torch.Tensor] = None,
        agent_valid_mask: Optional[torch.Tensor] = None,
    ):

        bs, num_agents, hidden_dim = x.shape
        base = self.input_proj(x).unsqueeze(2).unsqueeze(3)
        mode_embed = self.mode_embed.weight.view(1, 1, self.num_modes, 1, hidden_dim)
        hidden = base + mode_embed + self.time_embed
        hidden = self.temporal_decoder(hidden)
        hidden = hidden + self.channel_decoder(hidden)

        if self.use_cv_delta_prediction and (agent_current_state is not None):
            cv_base = self._build_cv_base_trajectory(
                agent_current_state=agent_current_state,
                target_agents=num_agents,
                target_dtype=hidden.dtype,
            ).unsqueeze(2).expand(-1, -1, self.num_modes, -1, -1)

            delta_loc = torch.cumsum(self.pos_predictor(hidden), dim=3)
            delta_yaw_raw = self.yaw_predictor(hidden)
            delta_yaw = torch.atan2(delta_yaw_raw[..., 1], delta_yaw_raw[..., 0]).unsqueeze(-1)
            delta_yaw = torch.cumsum(delta_yaw, dim=3)
            delta_vel = torch.cumsum(self.vel_predictor(hidden), dim=3)

            loc = cv_base[..., :2] + self.delta_scale * delta_loc
            yaw = cv_base[..., 2:3] + self.delta_scale * delta_yaw
            yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
            vel = cv_base[..., 3:4] + self.delta_scale * delta_vel
            prediction = torch.cat([loc, yaw, vel], dim=-1).contiguous()
        else:
            loc = self.pos_predictor(hidden)
            yaw = self.yaw_predictor(hidden)
            yaw = torch.atan2(yaw[..., 1], yaw[..., 0]).unsqueeze(-1)
            vel = self.vel_predictor(hidden)
            prediction = torch.cat([loc, yaw, vel], dim=-1).contiguous()

        if agent_valid_mask is not None:
            prediction_mask = agent_valid_mask.unsqueeze(2).unsqueeze(3).unsqueeze(4).to(dtype=prediction.dtype)
            prediction = prediction * prediction_mask

        confidence = self.confidence_predictor(hidden.mean(dim=3)).squeeze(-1)  # [B, N, M]
        if agent_valid_mask is not None:
            confidence = confidence.masked_fill(~agent_valid_mask.unsqueeze(-1), -30.0)

        return prediction, confidence


class AgentFutureEncoder(nn.Module):
    def __init__(self, dim: int, future_steps: int, num_modes: int) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.mode_encoder = Mlp(future_steps * 4, 2 * dim, dim)
        self.uncertainty_proj = nn.Linear(2, dim)
        self.fusion = nn.Sequential(
            nn.Linear(3 * dim, 2 * dim),
            LayerNorm(2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, traj: torch.Tensor, confidence_logits: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, num_agents, num_modes, num_steps, coord_dim = traj.shape
        mode_feat = self.mode_encoder(
            traj.reshape(batch_size * num_agents * num_modes, num_steps * coord_dim)
        ).view(batch_size, num_agents, num_modes, -1)

        if confidence_logits is None:
            mode_prob = traj.new_full((batch_size, num_agents, num_modes), 1.0 / num_modes)
        else:
            mode_prob = F.softmax(confidence_logits, dim=-1)
        weighted_feat = (mode_feat * mode_prob.unsqueeze(-1)).sum(dim=2)

        top1_idx = mode_prob.argmax(dim=-1)
        top1_feat = mode_feat.gather(
            2,
            top1_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, mode_feat.size(-1)),
        ).squeeze(2)

        max_prob = mode_prob.max(dim=-1, keepdim=True).values
        entropy = -(mode_prob.clamp_min(1e-8) * mode_prob.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        uncertainty_feat = self.uncertainty_proj(torch.cat([max_prob, entropy], dim=-1))
        return self.fusion(torch.cat([weighted_feat, top1_feat, uncertainty_feat], dim=-1))

class AnchorEncoder(nn.Module):
    def __init__(self, 
                 planner_anchor: torch.Tensor,
                 normalize_anchors: bool = False,
                 planning_horizon: float = 8.0,
                 d_model: int = 256,
                 encoding_depth: int = 10,
                 anchor_ts: int = 20):
        super().__init__()
        self.d_model = d_model
        self.normalize_anchors = normalize_anchors
        self.planning_horizon = planning_horizon
        ts = planner_anchor.shape[1]
        self.anchor_ts = ts
        self.encoding_depth = encoding_depth
        self.per_step_encoding_dim = 6 * encoding_depth

        self.pos_encoding = VectorizedFourierEncoding(L=encoding_depth)
        self.step_embedding = Mlp(
            self.per_step_encoding_dim,
            2 * d_model,
            d_model,
        )
        self.temporal_aggregator = nn.Sequential(
            nn.Linear(ts * d_model, 2 * d_model),
            LayerNorm(2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
    

    def forward(self, anchors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchors: [N, T, 6]
        Returns:
            encoded_anchors: [N, D]
        """
        if anchors.ndim != 3:
            raise ValueError(f"anchors must be [N, T, C], got shape={tuple(anchors.shape)}")
        if anchors.shape[1] != self.anchor_ts:
            raise ValueError(
                f"anchor length mismatch: expected T={self.anchor_ts}, got T={anchors.shape[1]}"
            )

        batch_anchor_num = anchors.shape[0]
        encoded_poses = self.pos_encoding(anchors[..., :3].contiguous())
        encoded_poses = encoded_poses.view(
            batch_anchor_num,
            self.anchor_ts,
            self.per_step_encoding_dim,
        )
        step_feat = self.step_embedding(encoded_poses.view(-1, self.per_step_encoding_dim))
        step_feat = step_feat.view(batch_anchor_num, self.anchor_ts, self.d_model)
        anchor_feat = self.temporal_aggregator(step_feat.reshape(batch_anchor_num, -1))

        return anchor_feat

class PlanningDecoderLayer(nn.Module):
    def __init__(self, 
                 d_model,
                 num_heads=8,
                 mlp_ratio=4.0,
                 layer_scale_init_value: float = 1e-4,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.ego_proj = nn.Linear(d_model, d_model)
        self.ego_gate = nn.Linear(d_model, d_model)
        self.cross_ref_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_env_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.ffn = Mlp(d_model, int(d_model * mlp_ratio), d_model)
        self.norm_ego = LayerNorm(d_model)
        self.norm_agent = LayerNorm(d_model)
        self.norm_env = LayerNorm(d_model)
        self.norm_ffn = LayerNorm(d_model)
        init = float(layer_scale_init_value)
        self.ego_inject_scale = nn.Parameter(torch.full((d_model,), init))
        self.agent_attn_scale = nn.Parameter(torch.full((d_model,), init))
        self.env_attn_scale = nn.Parameter(torch.full((d_model,), init))
        self.ffn_scale = nn.Parameter(torch.full((d_model,), init))

    def forward(self, 
                traj_feature, 
                ego_feature,
                agent_feature,
                agent_mask,
                env_feature,
                env_mask,
                ):
        x = traj_feature

        ego_ctx = self.ego_proj(ego_feature)
        ego_gate = torch.sigmoid(self.ego_gate(self.norm_ego(x)))
        x = x + self.dropout(self.ego_inject_scale * (ego_gate * ego_ctx))

        # Handle empty/all-masked keys to keep MultiheadAttention stable in sparse scenes.
        if agent_feature.size(1) == 0 or not bool(agent_mask.any()):
            agent_attn = torch.zeros_like(x)
        else:
            agent_attn = self.cross_ref_attention(
                self.norm_agent(x),
                agent_feature,
                agent_feature,
                key_padding_mask=~agent_mask,
                need_weights=False,
            )[0]
        x = x + self.dropout(self.agent_attn_scale * agent_attn)

        if env_feature.size(1) == 0 or not bool(env_mask.any()):
            env_attn = torch.zeros_like(x)
        else:
            env_attn = self.cross_env_attention(
                self.norm_env(x),
                env_feature,
                env_feature,
                key_padding_mask=~env_mask,
                need_weights=False,
            )[0]
        x = x + self.dropout(self.env_attn_scale * env_attn)
        x = x + self.dropout(self.ffn_scale * self.ffn(self.norm_ffn(x)))

        return x
    
class PredictionDecoderLayer(nn.Module):
    def __init__(self, 
                 d_model, 
                 nhead=8, 
                 mlp_ratio=4.0,
                 dropout=0.1,
                 layer_scale_init_value: float = 1e-4):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = Mlp(d_model, int(d_model * mlp_ratio), d_model)
        self.norm_q = LayerNorm(d_model)
        self.norm_ffn = LayerNorm(d_model)
        init = float(layer_scale_init_value)
        self.attn_scale = nn.Parameter(torch.full((d_model,), init))
        self.mlp_scale = nn.Parameter(torch.full((d_model,), init))

    def forward(self, agent_feat, agent_mask, scene_feat, scene_mask):
        
        x = torch.where(agent_mask.unsqueeze(-1), agent_feat, torch.zeros_like(agent_feat))

        if scene_feat.size(1) == 0 or not bool(scene_mask.any()):
            y = torch.zeros_like(x)
        else:
            y = self.cross_attn(
                self.norm_q(x),
                scene_feat,
                scene_feat,
                key_padding_mask=~scene_mask,
                need_weights=False,
            )[0]
        x = x + self.dropout(self.attn_scale * y)
        x = torch.where(agent_mask.unsqueeze(-1), x, torch.zeros_like(x))

        x = x + self.dropout(self.mlp_scale * self.ffn(self.norm_ffn(x)))
        x = torch.where(agent_mask.unsqueeze(-1), x, torch.zeros_like(x))
        return x
