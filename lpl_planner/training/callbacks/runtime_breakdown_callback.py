import logging
import time

import lightning.pytorch as pl
import torch


logger = logging.getLogger(__name__)


class RuntimeBreakdownCallback(pl.Callback):
    def __init__(
        self,
        warmup_steps: int = 20,
        log_every_n_steps: int = 100,
        synchronize_cuda: bool = True,
    ) -> None:
        super().__init__()
        self.warmup_steps = max(int(warmup_steps), 0)
        self.log_every_n_steps = max(int(log_every_n_steps), 1)
        self.synchronize_cuda = bool(synchronize_cuda)
        self._reset_epoch_state()

    def _reset_epoch_state(self) -> None:
        self._epoch_start_time = 0.0
        self._prev_batch_end_time = None
        self._batch_start_time = None
        self._current_data_wait = 0.0
        self._measured_steps = 0
        self._data_wait_total = 0.0
        self._step_total = 0.0

    def _sync_if_needed(self, trainer: pl.Trainer) -> None:
        if not self.synchronize_cuda:
            return
        root_device = getattr(trainer.strategy, "root_device", None)
        if isinstance(root_device, torch.device) and root_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(root_device)

    def _should_measure(self, trainer: pl.Trainer) -> bool:
        return int(trainer.global_step) >= self.warmup_steps

    def _maybe_log_running(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero or self._measured_steps <= 0:
            return
        if self._measured_steps % self.log_every_n_steps != 0:
            return

        avg_data_wait = self._data_wait_total / self._measured_steps
        avg_step = self._step_total / self._measured_steps
        wait_ratio = avg_data_wait / max(avg_data_wait + avg_step, 1e-8)
        logger.info(
            "Runtime profile step=%s avg_data_wait=%.4fs avg_train_step=%.4fs wait_ratio=%.1f%%",
            trainer.global_step,
            avg_data_wait,
            avg_step,
            100.0 * wait_ratio,
        )
        pl_module.log("profile/avg_data_wait_sec", avg_data_wait, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        pl_module.log("profile/avg_train_step_sec", avg_step, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        pl_module.log("profile/data_wait_ratio", wait_ratio, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._reset_epoch_state()
        self._sync_if_needed(trainer)
        now = time.perf_counter()
        self._epoch_start_time = now
        self._prev_batch_end_time = now

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx: int) -> None:
        self._sync_if_needed(trainer)
        now = time.perf_counter()
        if self._prev_batch_end_time is None:
            self._current_data_wait = 0.0
        else:
            self._current_data_wait = now - self._prev_batch_end_time
        self._batch_start_time = now

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        self._sync_if_needed(trainer)
        now = time.perf_counter()
        if self._batch_start_time is None:
            self._prev_batch_end_time = now
            return

        step_time = now - self._batch_start_time
        self._prev_batch_end_time = now

        if self._should_measure(trainer):
            self._measured_steps += 1
            self._data_wait_total += self._current_data_wait
            self._step_total += step_time
            self._maybe_log_running(trainer, pl_module)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero or self._measured_steps <= 0:
            return
        self._sync_if_needed(trainer)
        epoch_wall = time.perf_counter() - self._epoch_start_time
        avg_data_wait = self._data_wait_total / self._measured_steps
        avg_step = self._step_total / self._measured_steps
        measured_total = self._data_wait_total + self._step_total
        other_overhead = max(epoch_wall - measured_total, 0.0)
        logger.info(
            "Runtime profile epoch=%s measured_steps=%s avg_data_wait=%.4fs avg_train_step=%.4fs other_overhead=%.2fs",
            trainer.current_epoch,
            self._measured_steps,
            avg_data_wait,
            avg_step,
            other_overhead,
        )
