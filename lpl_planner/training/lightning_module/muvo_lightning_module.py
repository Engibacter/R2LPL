import lightning.pytorch as pl
import collections.abc as cabc
import torch
from typing import Dict, Tuple, Any, Optional
import logging

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)



class MVLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for reward model."""

    def __init__(self, model, 
                 cfg: Optional[DictConfig] = None,
                 learning_rate = 2e-4,
                 warmup_steps = 2000,
                 check_invalid_grad = False) -> None:
        """
        Initialise the lightning module wrapper.
        :param model: reward model
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.check_invalid_grad = check_invalid_grad
        self.cfg = cfg

    def _reduce_loss_value(self, value):
        if torch.is_tensor(value) and value.ndim > 0:
            return value.mean()
        return value

    def _move_to_device(self, obj, device):
        # NuPlan Feature/Target 对象
        if hasattr(obj, "to_device") and callable(getattr(obj, "to_device")):
            return obj.to_device(device)
        # Torch tensor
        if torch.is_tensor(obj):
            return obj.to(device)
        # Mapping / dict
        if isinstance(obj, cabc.Mapping):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}
        # List / Tuple
        if isinstance(obj, (list, tuple)):
            t = [self._move_to_device(v, device) for v in obj]
            return type(obj)(t) if not isinstance(obj, list) else t
        # 其他类型原样返回
        return obj

    def _sanitize_nonfinite_recursive(self, obj, prefix: str):
        if torch.is_tensor(obj):
            if obj.dtype == torch.bool:
                return obj
            finite = torch.isfinite(obj)
            if not finite.all():
                num_bad = int((~finite).sum().item())
                logger.warning("Tensor %s has %s non-finite values before/after forward. Sanitizing.", prefix, num_bad)
                if torch.is_floating_point(obj):
                    return torch.nan_to_num(obj, nan=0.0, posinf=1e6, neginf=-1e6)
                return torch.nan_to_num(obj)
            return obj

        if isinstance(obj, cabc.Mapping):
            return {key: self._sanitize_nonfinite_recursive(value, f"{prefix}.{key}") for key, value in obj.items()}

        if isinstance(obj, list):
            return [self._sanitize_nonfinite_recursive(value, f"{prefix}[{idx}]") for idx, value in enumerate(obj)]

        if isinstance(obj, tuple):
            return type(obj)(self._sanitize_nonfinite_recursive(value, f"{prefix}[{idx}]") for idx, value in enumerate(obj))

        if hasattr(obj, "__dict__"):
            for attr_name, attr_value in vars(obj).items():
                setattr(obj, attr_name, self._sanitize_nonfinite_recursive(attr_value, f"{prefix}.{attr_name}"))
        return obj
    
    def _step(self, batch: Tuple[Dict[str, Any], Dict[str, Any]], logging_prefix: str) -> torch.Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        device = self.device
        batch_size = features['scene_feature'].ego_feature.ego_current_state.shape[0]
        features = self._move_to_device(features, device)
        targets = self._move_to_device(targets, device)
        if getattr(self.model, "debug", False):
            features = self._sanitize_nonfinite_recursive(features, "batch.features")
            targets = self._sanitize_nonfinite_recursive(targets, "batch.targets")
        prediction = self.model.forward(features, targets)
        if getattr(self.model, "debug", False):
            prediction = self._sanitize_nonfinite_recursive(prediction, f"{logging_prefix}.prediction")
        # loss = self.agent.compute_loss(features, targets, prediction)
        # self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # return loss
        loss_dict = prediction['loss_dict']
        for k, v in loss_dict.items():
            if v is not None:
                self.log(
                    f"{logging_prefix}/{k}",
                    self._reduce_loss_value(v),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
        self.log(f"{logging_prefix}/mean_dist", prediction['mean_expert_distance'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        if logging_prefix == "train":
            self.log('learning_rate', self.optimizers().param_groups[0]['lr'], prog_bar=True)

        return self._reduce_loss_value(loss_dict['total_loss'])
    
    def on_after_backward(self):
        if self.check_invalid_grad or getattr(self.model, "debug", False):
            for name, param in self.model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    logger.warning("Gradient for %s contains NaN or Inf values!", name)

    def training_step(self, batch: Tuple[Dict[str, Any], Dict[str, Any]], batch_idx: int) -> torch.Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        
        
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Any], Dict[str, Any]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        
        if hasattr(self.trainer, "estimated_stepping_batches") and self.trainer.estimated_stepping_batches:
            total_steps = int(self.trainer.estimated_stepping_batches)
        elif self.trainer.max_steps and self.trainer.max_steps > 0:
            total_steps = int(self.trainer.max_steps)
        else:
            total_steps = 100000
        warmup_steps = max(self.warmup_steps, int(0.03 * total_steps))
        epoch_step = total_steps // self.trainer.max_epochs
        warmup_steps = max(warmup_steps, epoch_step+10)
        self.warmup_steps = warmup_steps

        # 线性 warmup（从 0.1% lr 提升到 base lr），然后余弦到 1e-5
        warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-5)
        seq = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
        
        # # epoch-wise LR reduction based on validation loss
        # plateau = ReduceLROnPlateau(
        #     optimizer,
        #     mode="min",          # minimize val loss
        #     factor=0.5,          # lr *= 0.5 on plateau
        #     patience=1,          # wait 1 epoch without improvement
        #     threshold=0.02,      # minimum improvement to count
        #     threshold_mode="rel",
        #     cooldown=0,          # cooldown epochs before next reduce
        #     min_lr=1e-5,
        # )

        # scheds = [
        #     {"scheduler": warmup, "interval": "step", "frequency": 1},
        #     {"scheduler": plateau, "interval": "epoch", "monitor": "val/total_loss_epoch" ,"frequency": 1},
        # ]
        # return [optimizer], scheds
        return [optimizer], [{"scheduler": seq, "interval": "step", "frequency": 1}]
    

    # def lr_scheduler_step(self, scheduler, metric):
    #     """
    #     - warmup 步内：只走 LinearLR，跳过 Plateau。
    #     - warmup 结束后：停止再调用 LinearLR，允许 Plateau 生效。
    #     """
    #     if isinstance(scheduler, ReduceLROnPlateau):
    #         if self.global_step < self.warmup_steps:
    #             return
    #         scheduler.step(metric)
    #     else:  # LinearLR
    #         if self.global_step < self.warmup_steps:
    #             scheduler.step()
    #         # warmup 结束后不再 step，避免覆盖 Plateau


    # def _nonfinite_state(self):
    #     issues = []
    #     with torch.no_grad():
    #         for name, tensor in list(self.model.named_parameters()) + list(self.model.named_buffers()):
    #             if tensor is None:
    #                 continue
    #             if not torch.isfinite(tensor).all():
    #                 n_nan = torch.isnan(tensor).sum().item()
    #                 n_inf = torch.isinf(tensor).sum().item()
    #                 issues.append((name, n_nan, n_inf))
    #     return issues

    # def on_train_start(self) -> None:
    #     issues = self._nonfinite_state()
    #     if issues:
    #         details = "; ".join(f"{name} (nan={n_nan}, inf={n_inf})" for name, n_nan, n_inf in issues)
    #         raise ValueError(f"Non-finite values detected before training: {details}")

        
