import logging
import os
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Union

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import importlib.util

from nuplan.common.utils.distributed_scenario_filter import DistributedMode, DistributedScenarioFilter
from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.script.builders.metric_builder import build_metrics_engines
from nuplan.planning.script.builders.observation_builder import build_observations
from nuplan.planning.script.builders.planner_builder import _build_planner
from nuplan.planning.script.builders.simulation_builder import build_simulations
from nuplan.planning.script.builders.simulation_callback_builder import (
    build_callbacks_worker,
    build_simulation_callbacks,
)
from nuplan.planning.script.builders.utils.utils_type import is_target_type
from nuplan.planning.script.utils import run_runners, set_default_path, set_up_common_builder
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from lpl_planner.utils.default_paths import configure_default_paths

import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
configure_default_paths()
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
nuplan_spec = importlib.util.find_spec("nuplan")
lpl_planner_spec = importlib.util.find_spec("lpl_planner")

nuplan_dir = os.path.dirname(nuplan_spec.origin) 
lpl_planner_dir = os.path.dirname(lpl_planner_spec.origin)
CONFIG_PATH = os.path.join(lpl_planner_dir, "config/simulation")

# if os.environ.get('NUPLAN_HYDRA_CONFIG_PATH') is not None:
#     CONFIG_PATH = os.path.join('../../../../', CONFIG_PATH)

# if os.path.basename(CONFIG_PATH) != 'simulation':
#     CONFIG_PATH = os.path.join(CONFIG_PATH, 'simulation')
CONFIG_NAME = 'default_simulation_ray'


class _LazyPlannerPlaceholder(AbstractPlanner):
    """Lightweight planner placeholder used only before worker-side construction."""

    def __init__(self, planner_name: str) -> None:
        self._planner_name = planner_name

    def name(self) -> str:
        return self._planner_name

    def initialize(self, initialization: PlannerInitialization) -> None:
        raise RuntimeError("Lazy planner placeholder cannot be initialized.")

    def observation_type(self):
        return DetectionsTracks

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        raise RuntimeError("Lazy planner placeholder cannot compute trajectories.")


class LazyPlannerSimulationRunner(SimulationRunner):
    """Simulation runner that instantiates the planner inside the execution worker."""

    def __init__(self, simulation: Simulation, planner_cfg: DictConfig, planner_key: str) -> None:
        planner_target = str(getattr(planner_cfg, "_target_", planner_key))
        planner_name = planner_target.rsplit(".", 1)[-1]
        super().__init__(simulation=simulation, planner=_LazyPlannerPlaceholder(planner_name))
        self._planner_cfg = planner_cfg
        self._planner_key = planner_key

    def run(self):
        """Build planner lazily in the worker process and run the native nuPlan simulation runner."""
        self._planner = _build_planner(self._planner_cfg, self.scenario)
        return super().run()


def build_simulations_lazy_planner(
    cfg: DictConfig,
    worker: WorkerPool,
    callbacks: List[AbstractCallback],
    callbacks_worker: Optional[WorkerPool] = None,
) -> List[LazyPlannerSimulationRunner]:
    """
    Build simulations without instantiating planners on the driver.

    This mirrors nuPlan's build_simulations, but stores each planner config in a
    LazyPlannerSimulationRunner so the heavy model/checkpoint/anchor allocation
    happens only inside the Ray task that executes the scenario.
    """
    logger.info('Building simulations with lazy worker-side planner construction...')
    simulations: List[LazyPlannerSimulationRunner] = []

    logger.info('Extracting scenarios...')
    if not int(os.environ.get("NUPLAN_SIMULATION_ALLOW_ANY_BUILDER", "0")) and not is_target_type(
        cfg.scenario_builder, NuPlanScenarioBuilder
    ):
        raise ValueError(f"Simulation framework only runs with NuPlanScenarioBuilder. Got {cfg.scenario_builder}")

    scenario_filter = DistributedScenarioFilter(
        cfg=cfg,
        worker=worker,
        node_rank=int(os.environ.get("NODE_RANK", 0)),
        num_nodes=int(os.environ.get("NUM_NODES", 1)),
        synchronization_path=cfg.output_dir,
        timeout_seconds=cfg.distributed_timeout_seconds,
        distributed_mode=DistributedMode[cfg.distributed_mode],
    )
    scenarios = scenario_filter.get_scenarios()

    metric_engines_map = {}
    if cfg.run_metric:
        logger.info('Building metric engines...')
        metric_engines_map = build_metrics_engines(cfg=cfg, scenarios=scenarios)
        logger.info('Building metric engines...DONE')
    else:
        logger.info('Metric engine is disabled')

    if 'planner' not in cfg.keys():
        raise KeyError('Planner not specified in config. Please specify a planner using "planner" field.')

    planner_items = list(cfg.planner.items())
    logger.info('Building lazy simulations from %d scenarios and %d planner config(s)...', len(scenarios), len(planner_items))

    for scenario in scenarios:
        for planner_key, planner_cfg in planner_items:
            ego_controller = instantiate(cfg.ego_controller, scenario=scenario)
            simulation_time_controller = instantiate(cfg.simulation_time_controller, scenario=scenario)
            observations = build_observations(cfg.observation, scenario=scenario)

            metric_engine = metric_engines_map.get(scenario.scenario_type, None)
            if metric_engine is not None:
                stateful_callbacks = [MetricCallback(metric_engine=metric_engine, worker_pool=callbacks_worker)]
            else:
                stateful_callbacks = []

            if "simulation_log_callback" in cfg.callback:
                stateful_callbacks.append(
                    instantiate(cfg.callback["simulation_log_callback"], worker_pool=callbacks_worker)
                )

            simulation_setup = SimulationSetup(
                time_controller=simulation_time_controller,
                observations=observations,
                ego_controller=ego_controller,
                scenario=scenario,
            )
            simulation = Simulation(
                simulation_setup=simulation_setup,
                callback=MultiCallback(callbacks + stateful_callbacks),
                simulation_history_buffer_duration=cfg.simulation_history_buffer_duration,
            )
            simulations.append(LazyPlannerSimulationRunner(simulation, planner_cfg, str(planner_key)))

    logger.info('Building lazy simulations...DONE!')
    return simulations


def run_simulation(cfg: DictConfig, planners: Optional[Union[AbstractPlanner, List[AbstractPlanner]]] = None) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Helper function for main to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    :param planners: Pre-built planner(s) to run in simulation. Can either be a single planner or list of planners.
    """
    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    profiler_name = 'building_simulation'
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)

    # Build simulation callbacks
    callbacks_worker_pool = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(cfg=cfg, output_dir=common_builder.output_dir, worker=callbacks_worker_pool)

    # Remove planner from config to make sure run_simulation does not receive multiple planner specifications.
    if planners and 'planner' in cfg.keys():
        logger.info('Using pre-instantiated planner. Ignoring planner in config')
        OmegaConf.set_struct(cfg, False)
        cfg.pop('planner')
        OmegaConf.set_struct(cfg, True)

    # Construct simulations
    if isinstance(planners, AbstractPlanner):
        planners = [planners]

    if planners is None:
        runners = build_simulations_lazy_planner(
            cfg=cfg,
            callbacks=callbacks,
            worker=common_builder.worker,
            callbacks_worker=callbacks_worker_pool,
        )
    else:
        runners = build_simulations(
            cfg=cfg,
            callbacks=callbacks,
            worker=common_builder.worker,
            pre_built_planners=planners,
            callbacks_worker=callbacks_worker_pool,
        )

    if common_builder.profiler:
        # Stop simulation construction profiling
        common_builder.profiler.save_profiler(profiler_name)

    logger.info('Running simulation...')
    run_runners(runners=runners, common_builder=common_builder, cfg=cfg, profiler_name='running_simulation')
    logger.info('Finished running simulation!')


def clean_up_s3_artifacts() -> None:
    """
    Cleanup lingering s3 artifacts that are written locally.
    This happens because some minor write-to-s3 functionality isn't yet implemented.
    """
    # Lingering artifacts get written locally to a 's3:' directory. Hydra changes
    # the working directory to a subdirectory of this, so we serach the working
    # path for it.
    working_path = os.getcwd()
    s3_dirname = "s3:"
    s3_ind = working_path.find(s3_dirname)
    if s3_ind != -1:
        local_s3_path = working_path[: working_path.find(s3_dirname) + len(s3_dirname)]
        rmtree(local_s3_path)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Calls run_simulation to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    """
    assert cfg.simulation_log_main_path is None, 'Simulation_log_main_path must not be set when running simulation.'
    # print(f"type(cfg.planner): {type(cfg.planner)}")
    # print(f"cfg.planner: {cfg.planner}")
    # Execute simulation with preconfigured planner(s).
    run_simulation(cfg=cfg)

    if is_s3_path(Path(cfg.output_dir)):
        clean_up_s3_artifacts()


if __name__ == '__main__':
    main()
