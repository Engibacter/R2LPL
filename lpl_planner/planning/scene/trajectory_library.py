import numpy as np
from enum import IntEnum
from scipy.stats import norm

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

import matplotlib.pyplot as plt
import pickle

import importlib.util
from tqdm import tqdm

from scipy.signal import savgol_filter

import hydra
from hydra.utils import instantiate
import glob
import os
import gc

lpl_planner_spec = importlib.util.find_spec("lpl_planner")
lpl_planner_dir = os.path.dirname(lpl_planner_spec.origin)
# Get relative paths based on package directories
CONFIG_PATH = os.path.join(lpl_planner_dir, "config", "training")
RESULT_PATH = os.path.join(os.environ.get("R2LPL_RESULTS_ROOT", os.path.join(lpl_planner_dir, "..", "results")), "logs")
CONFIG_NAME = "trajectory_library"
LIBRARY_PATH = os.path.join(RESULT_PATH, "trajectory_library")

class TrajectoryState(IntEnum):
    X = 0
    Y = 1
    HEADING = 2
    VELOCITY_X = 3
    VELOCITY_Y = 4
    ACCELERATION_X = 5
    ACCELERATION_Y = 6
    YAW_RATE = 7
    YAW_ACCELERATION = 8
    JERK_X = 9
    JERK_Y = 10


    # Define slices for different parts of the trajectory state
    # POINT = slice(X, Y + 1)
    # STATE = slice(X, HEADING + 1)
    # VELOCITY = slice(VELOCITY_X, VELOCITY_Y + 1)
    # ACCELERATION = slice(ACCELERATION_X, ACCELERATION_Y + 1)

    def size() -> int:
        """
        Returns the number of elements in the trajectory state.
        :return: The number of elements in the trajectory state.
        """
        return 11
    
    @classmethod
    def POINT(cls):
        """
        Returns the slice for the point part of the trajectory state.
        """
        return slice(cls.X, cls.Y + 1)

    @classmethod
    def STATE(cls):
        """
        Returns the slice for the state part of the trajectory state.
        """
        return slice(cls.X, cls.HEADING + 1)

    @classmethod
    def VELOCITY(cls):
        """
        Returns the slice for the velocity part of the trajectory state.
        """
        return slice(cls.VELOCITY_X, cls.VELOCITY_Y + 1)

    @classmethod
    def ACCELERATION(cls):
        """
        Returns the slice for the acceleration part of the trajectory state.
        """
        return slice(cls.ACCELERATION_X, cls.ACCELERATION_Y + 1)


def phase_unwrap(headings):
    """
    Returns an array of heading angles equal mod 2 pi to the input heading angles,
    and such that the difference between successive output angles is less than or
    equal to pi radians in absolute value
    :param headings: An array of headings (radians)
    :return The phase-unwrapped equivalent headings.
    """
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[1:] = np.cumsum(np.round(np.diff(headings) / two_pi))
    unwrapped = headings - two_pi * adjustments
    return unwrapped

class TrajectoryLibrary:
    def __init__(self, 
                 library_path=LIBRARY_PATH, 
                 time_horizon=80,
                 time_interval=0.1,
                 load_trajectories=True):
        self.library_path = library_path
        self.time_horizon = time_horizon
        self.time_interval = time_interval
        self.library_index = {} #{file_name: [scenario_token]}
        self.state_index = {} #{file_name: [velocity, acceleration]}
        self.trajectories = None  # Loaded on demand
        self.load_trajectories = load_trajectories
        self._load_library()

        # Check if the trajectory array is created
        if not self.library_index:
            raise RuntimeError("Trajectory library is empty or not loaded correctly.")

    def _load_library(self):
        """
        Loads the trajectory library from the specified path.
        """
        
        part_files = sorted(glob.glob(f"{self.library_path}/trajectory_library_part_*.pkl"))
        if not part_files:
            raise RuntimeError(f"No trajectory library part files found in {self.library_path}.")
        
        self.library_index = {}
        self.trajectories = []
        for part_file in part_files:
            with open(part_file, "rb") as f:
                data = pickle.load(f)
            for scenario_token, traj_list in data.items():
                if part_file not in self.library_index:
                    self.library_index[part_file] = []
                    self.state_index[part_file] = []
                self.library_index[part_file].append(scenario_token)
                self.state_index[part_file].append([traj_list[0][TrajectoryState.VELOCITY_X], traj_list[0][TrajectoryState.ACCELERATION_X]])
                self.trajectories.extend(traj_list)
        # concate state index for better reference speed
        self.state_index = {k: np.array(v) for k, v in self.state_index.items()}
        self.trajectories = np.stack(self.trajectories)  # Shape: (N, H, 11)
        # Free memory by deleting loaded data and running garbage collection
        del data
        gc.collect()
        # print("trajectory library loaded")

    def get_trajectory(self, scenario_token, run_step=0, time_horizon=8):
        """
        Returns the trajectory ndarray for the given scenario token and run step.
        """
        # Find the part file containing the scenario token
        for part_file, scenarios in self.library_index.items():
            if scenario_token in scenarios:
                with open(part_file, "rb") as f:
                    data = pickle.load(f)
                    traj = data[scenario_token][run_step]
        return traj[:int(time_horizon / self.time_interval + 1),:]
    
    def sample_with_state(self, 
                          max_samples=10, 
                          time_horizon=8, 
                          ego_state:EgoState=None, 
                          velocity_tolerance=0.5, 
                          acceleration_tolerance=1.0, 
                          ego_traj=None,
                          max_search_sample=2000,
                          negative_sample_ratio=0.5,
                          positive_sample_method='random'):
        """
        Randomly samples a number of trajectories from all part files starting from the given ego state.
        :param max_samples: Maximum number of trajectories to sample.
        :param time_horizon: Time horizon for the trajectories. [s]
        :param ego_state: Current state of the ego vehicle. [EgoState]
        :param velocity_tolerance: Tolerance for matching velocity.
        :param acceleration_tolerance: Tolerance for matching acceleration.
        :param ego_traj: Expert trajectory for diversity sampling.
        :param max_search_sample: Maximum number of candidate samples to consider before selecting final samples.
        :param negative_sample_ratio: Ratio of negative samples to positive samples when ego_traj is provided.
        :param positive_sample_method: Method to select positive samples, either 'similar' or 'random' or 'uniform'.
        """
        if ego_state is None:
            raise ValueError("Ego state must be provided.")

        if ego_traj is not None and not isinstance(ego_traj, np.ndarray):
            raise ValueError("Ego trajectory must be a numpy array.")   
        
        desired_velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        desired_acceleration = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x

        samples = []

        # Find candidate trajectories based on velocity and acceleration tolerance
        mask = (
            (np.abs(self.trajectories[:, 0, TrajectoryState.VELOCITY_X] - desired_velocity) <= velocity_tolerance) &
            (np.abs(self.trajectories[:, 0, TrajectoryState.ACCELERATION_X] - desired_acceleration) <= acceleration_tolerance)
        )
        samples = [self.trajectories[i][:int(time_horizon / self.time_interval + 1), :] for i in np.where(mask)[0]]
                
        # If too many samples, randomly select max_search_sample
        if len(samples) > max_search_sample:
            indices = np.random.choice(len(samples), max_search_sample, replace=False)
            samples = [samples[i] for i in indices]
        

        # If ego_traj is provided, select a mix of positive and negative samples
        if ego_traj is not None and len(samples) > 0:
            
            num_negative = min(int(max_samples * negative_sample_ratio), len(samples))
            num_positive = min(max_samples - num_negative, len(samples) - num_negative)

            # Negative samples: most different
            # greedy selection based on max-min distance
            selected = [samples[0]]
            selected_indices = [0]
            for _ in range(1, num_negative):
                min_dists = []
                for i, traj in enumerate(samples):
                    if i in selected_indices:
                        min_dists.append(-np.inf)
                        continue
                    dists = [np.linalg.norm(traj[:,:3] - s[:,:3]) for s in selected]
                    min_dists.append(np.min(dists))
                next_idx = np.argmax(min_dists)
                selected.append(samples[next_idx])
                selected_indices.append(next_idx)
            negative_samples = selected

            # Positive samples: most similar
            # Compute distances to ego_traj
            dists = np.array([np.linalg.norm(traj[:,:3] - ego_traj[:,:3]) for traj in samples])
            remaining_indices = np.array([i for i in range(len(samples)) if i not in selected_indices],dtype=int)
            remaining_dists = dists[remaining_indices]
            sorted_remaining_indices = [remaining_indices[i] for i in np.argsort(remaining_dists)]
            if num_positive > 0:
                if positive_sample_method == 'random':
                    top_n = sorted_remaining_indices[:max(num_positive, len(sorted_remaining_indices)//2)]
                    chosen = np.random.choice(top_n, num_positive, replace=False)
                    positive_samples = [samples[i] for i in chosen]
                elif positive_sample_method == 'similar':
                    positive_indices = sorted_remaining_indices[:num_positive]
                    positive_samples = [samples[i] for i in positive_indices]
                elif positive_sample_method == 'uniform':
                    top_n = sorted_remaining_indices[:max(num_positive, len(sorted_remaining_indices)//2)]
                    step = max(1, len(top_n) // num_positive)
                    positive_samples = [samples[top_n[i]] for i in range(0, len(top_n), step)][:num_positive]
                else:
                    raise ValueError(f"Unknown positive_sample_method: {positive_sample_method}")
            else:
                positive_samples = []

            negative_samples = np.stack(negative_samples) if len(negative_samples) > 0 else None
            positive_samples = np.stack(positive_samples) if len(positive_samples) > 0 else None


            return negative_samples, positive_samples
        else:
            return None, None 


def get_trajectory_from_scenario(scenario: AbstractScenario, 
                                 run_step=0, 
                                 time_horizon=8, 
                                 time_interval=0.1):
    """
    Returns the trajectory ndarray for the given scenario.
    """

    num_samples = int(time_horizon / time_interval + 1)

    ego_state = scenario.get_ego_state_at_iteration(run_step)
            
    expert_traj = scenario.get_ego_future_trajectory(iteration=run_step,
                                                    time_horizon=time_horizon,
                                                    num_samples=int(time_horizon / time_interval),)
    expert_traj = [ego_state for ego_state in expert_traj]


    # Initialize the trajectory arrays
    expert_acc = np.zeros((num_samples, 2))
    expert_velocity = np.zeros((num_samples, 2))
    expert_steering_rate = np.zeros((num_samples, 1))
    expert_headings = np.zeros((num_samples, ))
    expert_xy = np.zeros((num_samples, 2))

    # Initialize the first state
    expert_acc[0,0] = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
    expert_acc[0,1] = ego_state.dynamic_car_state.rear_axle_acceleration_2d.y
    expert_velocity[0,0] = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
    expert_velocity[0,1] = ego_state.dynamic_car_state.rear_axle_velocity_2d.y  
    expert_steering_rate[0,0] = ego_state.dynamic_car_state.tire_steering_rate
    expert_headings[0] = ego_state.rear_axle.heading
    expert_xy[0,0] = ego_state.rear_axle.x
    expert_xy[0,1] = ego_state.rear_axle.y

    # Fill the trajectory arrays with the expert trajectory data
    for idx, ego_state in enumerate(expert_traj): 

        expert_acc[idx+1,0] = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
        expert_acc[idx+1,1] = ego_state.dynamic_car_state.rear_axle_acceleration_2d.y
        expert_velocity[idx+1,0] = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        expert_velocity[idx+1,1] = ego_state.dynamic_car_state.rear_axle_velocity_2d.y
        expert_steering_rate[idx+1,0] = ego_state.dynamic_car_state.tire_steering_rate
        expert_headings[idx+1] = ego_state.rear_axle.heading
        expert_xy[idx+1,0] = ego_state.rear_axle.x
        expert_xy[idx+1,1] = ego_state.rear_axle.y

    # convert to local coordinates
    init_yaw = - expert_headings[0]
    rotation_matrix = np.array([
        [np.cos(init_yaw), -np.sin(init_yaw)],
        [np.sin(init_yaw), np.cos(init_yaw)]
    ])

    expert_local_xy = np.dot(expert_xy - expert_xy[0], rotation_matrix.T)
    # print(f"expert_headings.shape: {expert_headings.shape}")
    expert_local_headings = phase_unwrap(expert_headings - expert_headings[0])

    filtered_acceleration_x = savgol_filter(
        expert_acc[:,0], polyorder=2, window_length=min(8, len(expert_acc))
    )
    filtered_acceleration_y = savgol_filter(
        expert_acc[:,1], polyorder=2, window_length=min(8, len(expert_acc))
    )
    filtered_acceleration_x = np.round(filtered_acceleration_x, decimals=8)
    filtered_acceleration_y = np.round(filtered_acceleration_y, decimals=8)

    time_point = np.array([ego_state.time_point.time_us for ego_state in expert_traj])

    dt = np.diff(time_point/1e6).mean()

    yaw_rate = savgol_filter(
        expert_local_headings, polyorder=2, window_length=min(8, len(expert_local_headings)), deriv=1, delta=dt, axis=0
    )

    yaw_acceleration = savgol_filter(
        expert_local_headings, polyorder=3, window_length=min(8, len(expert_local_headings)), deriv=2, delta=dt, axis=0
    )
    
    jerk_x = savgol_filter(
        filtered_acceleration_x, polyorder=2, window_length=min(8, len(filtered_acceleration_x)), deriv=1, delta=dt, axis=0
    )

    jerk_y = savgol_filter(
        filtered_acceleration_y, polyorder=2, window_length=min(8, len(filtered_acceleration_y)), deriv=1, delta=dt, axis=0
    )

    trajectory = np.stack([
        expert_local_xy[:,0], 
        expert_local_xy[:,1],
        expert_local_headings,
        expert_velocity[:,0], 
        expert_velocity[:,1],
        filtered_acceleration_x, 
        filtered_acceleration_y,
        yaw_rate, 
        yaw_acceleration,
        jerk_x, 
        jerk_y
    ], axis=-1)
    
    return trajectory.astype(np.float32)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def process(cfg):
    
    # pytorch_seed.seed(2)
    # ch = cache.LocalCache("mppi_res.pkl")

    # get scenarios
    scenario_builder = instantiate(cfg.scenario_builder)
    # scenario_filter = ScenarioFilter(*get_filter_parameters(num_scenarios_per_type=None, limit_total_scenarios=1, shuffle=False))
    scenario_filter = instantiate(cfg.scenario_filter)
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = scenario_builder.get_scenarios(scenario_filter, worker)

    time_horizon = cfg.time_horizon # seconds
    time_interval = cfg.time_step # seconds
    num_samples = int(time_horizon / time_interval)
    

    traj_counter = 0
    scene_counter = 0
    print('Number of scenarios:', len(scenarios))
    data = {}

    for scenario in tqdm(scenarios):
        run_steps = scenario.get_number_of_iterations()
        token = scenario.token
        trajectories = []
        for i in range(run_steps):
            
            traj_counter += 1

            trajectory = get_trajectory_from_scenario(scenario, run_step=i, time_horizon=time_horizon, time_interval=time_interval)

            # print(f"trajectory.shape: {trajectory.shape}")

            trajectories.append(trajectory)
        
        
        scene_counter += 1
        data[token] = trajectories
        # Save after every 10k trajectories.
        if traj_counter % 10000 == 0 and traj_counter > 0:
            part_path = os.path.join(LIBRARY_PATH, f"trajectory_library_part_{traj_counter//10000}.pkl")
            os.makedirs(os.path.dirname(part_path), exist_ok=True)
            with open(part_path, "wb") as f:
                pickle.dump(data, f)
                print(f"Saved {traj_counter} trajectories to {part_path}")
                data = {}  # Clear the buffer for the next file.

    print(f"Processed {scene_counter} scenarios with {traj_counter} trajectories.")

if __name__ == "__main__":
    process()
