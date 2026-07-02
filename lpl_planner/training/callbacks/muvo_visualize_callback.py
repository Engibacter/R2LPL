import torch
import torchvision.utils as vutils
import collections.abc as cabc
import lightning.pytorch as pl
import gc
import numpy as np
from lpl_planner.training.dataset.utils import draw_model_in_out
from lpl_planner.planning.scene.scene_feature.features import SceneFeature

class MVVisualizeCallback(pl.Callback):

    def __init__(self,
                num_plots: int = 2,
                num_rows: int = 1,
                num_columns: int = 1,
        ) -> None:
        super().__init__()
        self._num_plots = num_plots
        self._num_rows = num_rows
        self._num_columns = num_columns
        self._cached_train_batch = None
        self._cached_val_batch = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._cached_train_batch = None
        self._cached_val_batch = None
    
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx) -> None:
        # Cache the first training batch for this epoch's visualization.
        if batch_idx == 0 and self._cached_train_batch is None:
            self._cached_train_batch = batch
    
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        # Cache the first validation batch for this epoch's visualization.
        if batch_idx == 0 and dataloader_idx == 0 and self._cached_val_batch is None:
            self._cached_val_batch = batch

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._cached_train_batch is None:
            return
        
        # Visualize the model's predictions on the training set
        device = pl_module.device
        features, targets = self._cached_train_batch
        features = self._move_to_device(features, device)

        was_training = pl_module.model.training
        pl_module.model.eval()
        scene_feature = features['scene_feature']
        scene_feature_unpacked = scene_feature.unpack()
        scene_feature_sampled = scene_feature_unpacked[:self._num_plots]
        features['scene_feature'] = SceneFeature.collate(scene_feature_sampled).to_device(device)
        with torch.no_grad():
            predictions = pl_module.model.sample_trajectories(features,
                                                              temperature=1.0,
                                                              num_samples=5,
                                                              max_chunk_num=1,
                                                              top_k=5)
        if was_training:
            pl_module.model.train()
        
        trajectories = predictions['trajectories'].detach().cpu().numpy()
        expert_trajectories = targets['expert_trajectory'].data.detach().cpu().numpy()
        all_trajectory_scores = predictions['scores'].detach().cpu().numpy()
        agent_prediction_gt = targets['agent_prediction']
        agent_prediction = predictions.get('agent_prediction')
        agent_prediction_gt = agent_prediction_gt.to_device(torch.device("cpu")).unpack()
        if agent_prediction is not None:
            agent_prediction = agent_prediction.to_device(torch.device("cpu")).unpack()
        scene_feature = scene_feature.to_device(torch.device("cpu")).unpack()
        gird_size = self._num_rows * self._num_columns
        num_scenes = self._num_plots*gird_size
        batch_size = trajectories.shape[0]

        if num_scenes > batch_size:
            num_plot = batch_size // gird_size
        else:
            num_plot = self._num_plots

        for idx_plot in range(num_plot):
            plots = []
            for sample_idx in np.arange(gird_size*idx_plot, gird_size*(idx_plot+1)):
                img = draw_model_in_out(
                    scene_feature=scene_feature[sample_idx],
                    expert_trajectory=expert_trajectories[sample_idx],
                    chosen_trajectory=trajectories[sample_idx][0],  # first trajectory as chosen
                    all_trajectories=trajectories[sample_idx][1:],  # skip the first trajectory 
                    all_trajectory_scores=all_trajectory_scores[sample_idx][1:],
                    agent_prediction=agent_prediction[sample_idx] if agent_prediction is not None else None,
                    agent_prediction_gt=agent_prediction_gt[sample_idx]
                )
                plots.append(torch.tensor(img).permute(2,0,1))  # HWC to CHW
            grid = vutils.make_grid(plots, normalize=False, nrow=self._num_rows)
            trainer.logger.experiment.add_image(f"train_plot_{idx_plot}", grid, global_step=trainer.current_epoch)
        
        # Clear cached batch after use
        self._cached_train_batch = None
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()

    def on_validation_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        if self._cached_val_batch is None:
            return

        # Visualize the model's predictions on the training set
        device = lightning_module.device
        features, targets = self._cached_val_batch
        features = self._move_to_device(features, device)

        was_training = lightning_module.model.training
        lightning_module.model.eval()

        with torch.no_grad():
            predictions = lightning_module.model.sample_trajectories(features,
                                                                     temperature=1.0,
                                                                     top_k=5)
        if was_training:
            lightning_module.model.train()

        
        scene_feature = features['scene_feature']
        trajectories = predictions['trajectories'].detach().cpu().numpy()
        expert_trajectories = targets['expert_trajectory'].data.detach().cpu().numpy()
        all_trajectory_scores = predictions['scores'].detach().cpu().numpy()
        scene_feature = scene_feature.to_device(torch.device("cpu")).unpack()
        agent_prediction = predictions.get('agent_prediction')
        if agent_prediction is not None:
            agent_prediction = agent_prediction.to_device(torch.device("cpu")).unpack()
        agent_prediction_gt = targets['agent_prediction']
        agent_prediction_gt = agent_prediction_gt.to_device(torch.device("cpu")).unpack()
        gird_size = self._num_rows * self._num_columns
        num_scenes = self._num_plots*gird_size
        batch_size = trajectories.shape[0]

        if num_scenes > batch_size:
            num_plot = batch_size // gird_size
        else:
            num_plot = self._num_plots

        for idx_plot in range(num_plot):
            plots = []
            for sample_idx in np.arange(gird_size*idx_plot, gird_size*(idx_plot+1)):
                img = draw_model_in_out(
                    scene_feature=scene_feature[sample_idx],
                    expert_trajectory=expert_trajectories[sample_idx],
                    chosen_trajectory=trajectories[sample_idx][0],  # first trajectory as chosen
                    all_trajectories=trajectories[sample_idx][1:],  # skip the first trajectory
                    all_trajectory_scores=all_trajectory_scores[sample_idx][1:],
                    agent_prediction=agent_prediction[sample_idx] if agent_prediction is not None else None,
                    agent_prediction_gt=agent_prediction_gt[sample_idx]
                )
                plots.append(torch.tensor(img).permute(2,0,1))  # HWC to CHW
            grid = vutils.make_grid(plots, normalize=False, nrow=self._num_rows)
            trainer.logger.experiment.add_image(f"val_plot_{idx_plot}", grid, global_step=trainer.current_epoch)
        
        # Clear cached batch after use
        self._cached_val_batch = None
        # print(f"Validation plots logged for epoch {trainer.current_epoch}.")

    def _move_to_device(self, obj, device):
        # nuPlan feature/target objects
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
        # Leave other object types unchanged.
        return obj
