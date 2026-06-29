from typing import Optional
from pathlib import Path
import os
import logging
import importlib.util
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from datetime import timedelta

# 开启 Tensor Cores 的 TF32 路径以加速 float32 matmul
import torch
torch.set_float32_matmul_precision('high')  # 或 'medium'

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from lpl_planner.training.dataset.mv_dataset import MVDataset
from lpl_planner.training.dataset.dataset_utils import FeatureCollate
from lpl_planner.training.lightning_module.muvo_lightning_module import MVLightningModule
from lpl_planner.training.callbacks.muvo_visualize_callback import MVVisualizeCallback
from lpl_planner.training.callbacks.runtime_breakdown_callback import RuntimeBreakdownCallback
from lpl_planner.utils.default_paths import configure_default_paths
from glob import glob

configure_default_paths()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

nuplan_spec = importlib.util.find_spec("nuplan")
hybrid_planner_spec = importlib.util.find_spec("hybrid_planner")
nuplan_dir = os.path.dirname(nuplan_spec.origin) 
hybrid_planner_dir = os.path.dirname(hybrid_planner_spec.origin)

CONFIG_PATH = os.path.join(hybrid_planner_dir, "config/training")
CONFIG_NAME = "custom_mv_training_server"

def _find_resume_ckpt(ckpt_dir: str) -> Optional[str]:
    
    last = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last):
        return last
    cands = glob(os.path.join(ckpt_dir, "*.ckpt"))
    return max(cands, key=os.path.getmtime) if cands else None

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    disable_checkpointing = bool(getattr(cfg, "disable_checkpointing", False))
    val_ratio = float(getattr(cfg.dataset, "val_ratio", 0.1))
    val_split_seed = int(getattr(cfg.dataset, "val_split_seed", 0))
    use_manifest = bool(getattr(cfg.cache, "use_manifest", False))
    scene_tokens_path = getattr(cfg.cache, "scene_tokens_path", None) if use_manifest else None

    if cfg.train_data == 'all':
        train_log_names = None
    elif cfg.train_data == 'train_only':
        train_log_names = cfg.splitter.log_splits.train
    elif cfg.train_data == 'val_only':
        train_log_names = cfg.splitter.log_splits.val
    elif cfg.train_data == 'test_only':
        train_log_names = cfg.splitter.log_splits.test
    elif cfg.train_data == 'train_val':
        train_log_names = cfg.splitter.log_splits.train + cfg.splitter.log_splits.val
    else:
        raise ValueError(f"Invalid train_data option: {cfg.train_data}")
    
    train_dataset_kwargs = dict(
        future_sampling=instantiate(cfg.model.future_sampling),
        scene_tokens_path=scene_tokens_path,
        use_anchor_indice=getattr(cfg.cache, "use_anchor_indice", False),
        use_anchor_score=getattr(cfg.cache, "use_anchor_score", False),
        anchor_indice_name=getattr(cfg.cache, "anchor_indice_name", None),
        anchor_score_name=getattr(cfg.cache, "anchor_score_name", None),
        use_factorized_anchor_target=getattr(cfg.cache, "use_factorized_anchor_target", False),
        use_manifest=use_manifest,
    )

    logger.info(f"Train dataset kwargs: {train_dataset_kwargs}")
    train_dataset = MVDataset(
        cache_path=Path(cfg.cache.cache_path),
        log_names=train_log_names, 
        expand_iteration=getattr(cfg.cache, "expand_iteration", False),
        **train_dataset_kwargs,
    )   
    # train_dataset_plus = MVDataset(
    #     cache_path=Path(os.environ["R2LPL_CACHE_ROOT"]) / "test_rl_val14_caching",
    #     log_names=None,
    #     expand_iteration=True,
    #     **train_dataset_kwargs,
    # )
    # train_dataset.add_dataset(train_dataset_plus)

    val_dataset = train_dataset.split_val_dataset(
        val_ratio=val_ratio,
        random_seed=val_split_seed,
    )
    # test_dataset = RMDataset(
    #     cache_path=Path(cfg.cache.cache_path),
    #     sample_num=cfg.dataset.sample_per_scenario,
    #     sample_ratio=cfg.dataset.negative_sample_ratio,
    #     log_names=cfg.splitter.log_splits.test,
    # )
    print(f"Number of training samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Number of validation samples: {len(val_dataset)}")
    else:
        print("Validation split disabled")
    # print(f"Number of test samples: {len(test_dataset)}")

    train_dataloader = DataLoader(dataset=train_dataset,
                                  drop_last=True,
                                  shuffle=True,
                                  collate_fn=FeatureCollate(),
                                  **cfg.dataloader.params, 
                                  )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(dataset=val_dataset,
                                    drop_last=True,
                                    shuffle=False,
                                    collate_fn=FeatureCollate(),
                                    **cfg.val_dataloader.params,
                                    )
    print(f'model config: {cfg.model}')
    model = instantiate(cfg.model)
    # print(f"Model structure: {model}")
    rm_lightning_module = MVLightningModule(model, 
                                            learning_rate=cfg.learning_rate,
                                            warmup_steps=cfg.warmup_steps,
                                            check_invalid_grad=cfg.check_invalid_grad)
    logger.info("Building Trainer")
    ckpt_dir = f"results/checkpoints/{cfg.job_name}"
    
    # construct callbacks
    visualize_callback = MVVisualizeCallback()
    runtime_breakdown_callback = RuntimeBreakdownCallback(
        warmup_steps=int(getattr(cfg, "profile_runtime_warmup_steps", 20)),
        log_every_n_steps=int(getattr(cfg, "profile_runtime_log_every_n_steps", 100)),
        synchronize_cuda=bool(getattr(cfg, "profile_runtime_sync_cuda", True)),
    )

    # ckpt_val_best = ModelCheckpoint(
    #     dirpath=ckpt_dir,
    #     filename="val-best-{epoch:02d}-{val_loss:.4f}",
    #     save_top_k=1,
    #     monitor="val/total_loss",
    #     mode="min",
    # )

    ckpt_train_best = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="train-best-{epoch:02d}-{train_loss:.4f}",
        save_top_k=1,
        monitor="train/total_loss",
        mode="min",
    )
    ckpt_last = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="last",
        save_last=True,     # 每个 epoch 结束保存 last.ckpt
        monitor=None,       # 不监控指标，仅保存最新
        save_top_k=0,
    )
    # Save a checkpoint every n epochs (also keep a rolling last.ckpt)
    ckpt_n_epoch = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch-{epoch:03d}",
        monitor=None,
        save_top_k=-1,                # keep all periodic checkpoints
        save_last=True,               # also keep last.ckpt
        every_n_epochs=10,
        save_on_train_epoch_end=True, # save at end of train epoch
    )
    # rm_lightning_module = torch.compile(rm_lightning_module)

    
    os.makedirs(ckpt_dir, exist_ok=True)
    
    callbacks = [] if disable_checkpointing else [ckpt_train_best, ckpt_last]
    if disable_checkpointing:
        logger.info("Checkpointing disabled by cfg.disable_checkpointing=True")

    if bool(getattr(cfg, "profile_runtime", False)):
        callbacks.append(runtime_breakdown_callback)

    if cfg.visualize:
        callbacks.append(visualize_callback)
        
    resume_ckpt = None if disable_checkpointing else _find_resume_ckpt(ckpt_dir)
    
    if cfg.start_model_path is not None and cfg.start_model_path != "None":
        start_ckpt = Path(cfg.start_model_path)
        logger.info(f"Starting from model checkpoint (weights-only): {start_ckpt}")
        ckpt = torch.load(start_ckpt, map_location=torch.device("cpu"))
        # 仅提取模型权重；Lightning 保存的键通常以 "model." 开头
        state = ckpt.get("state_dict", ckpt)

        def strip_prefix(state_dict, prefix):
            plen = len(prefix)
            return {k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)}
        
        sub_state = {}
        for p in ['model.', 'policy.']:
            stripped_state = strip_prefix(state, p)
            if len(stripped_state) > 0:
                prefix = p
                sub_state = stripped_state
                break
        if not sub_state:
            raise KeyError(f"No matching keys for prefixes {['model.', 'policy.']}. Available top-level keys (sample): "
                        f"{list(state.keys())[:5]}"
                        )
        state_dict = {k.replace(prefix, ""): v for k, v in sub_state.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Weights loaded: missing={len(missing)}, unexpected={len(unexpected)} from {start_ckpt}")
        # 关键：不恢复训练进展，禁止 Trainer.fit 使用 ckpt_path
        resume_ckpt = None
    elif cfg.pretrained_path is not None and cfg.pretrained_path != "None":
       pretrained_ckpt = Path(cfg.pretrained_path)
       ckpt = torch.load(pretrained_ckpt, map_location=torch.device("cpu"))
       raw = ckpt["state_dict"]
       state_dict = {k.replace("model.", ""): v for k, v in raw.items()}

       # 只允许这些前缀（按你的模型命名调整：state_encoder/agent_decoder/type_embed/prediction_head）
       allowed_prefixes = ("state_encoder.", "agent_decoder.", "type_embed.", "prediction_head.")

       model_sd = model.state_dict()
       partial_sd = {}
       for k, v in state_dict.items():
           if k.startswith(allowed_prefixes) and (k in model_sd) and (model_sd[k].shape == v.shape):
               partial_sd[k] = v

       # 关键：只加载 partial_sd，避免尺寸不匹配键
       missing, unexpected = model.load_state_dict(partial_sd, strict=False)
       logger.info(f"Pretrained partial load: {len(partial_sd)} tensors loaded; "
                   f"missing={len(missing)}, unexpected={len(unexpected)} from {pretrained_ckpt}")
       logger.info(f"Pretrained checkpoint specified, using: {pretrained_ckpt}")
    
    has_unused_param =  bool(getattr(cfg.model, "use_moe_decoder", False))
    trainer = pl.Trainer(
            strategy=DDPStrategy(
            find_unused_parameters=has_unused_param,       
            gradient_as_bucket_view=True,
            static_graph=False,                
            timeout=timedelta(minutes=10),
        ),
        callbacks=callbacks,
        **cfg.lightning.trainer.params
    )
    if resume_ckpt:
        logger.info(f"Resuming from checkpoint: {resume_ckpt}")
    else:
        logger.info("No checkpoint found. Starting fresh.")

    # from lightning.pytorch.tuner import Tuner
    # tuner = Tuner(trainer)
    # new_bs = tuner.scale_batch_size(
    #     rm_lightning_module,
    #     train_dataloaders=train_dataloader,
    #     mode="binsearch",   # 或 "power"
    #     init_val=8,
    #     max_trials=25,
    # )
    # print("Suggested batch size:", new_bs)

    logger.info("Starting Training")
    trainer.fit(rm_lightning_module,
                train_dataloader,
                val_dataloader,
                ckpt_path=resume_ckpt,
                )
    
if __name__ == "__main__":
    main()
