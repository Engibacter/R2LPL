import torch
import numpy as np
from enum import IntEnum

class STATEINDEX(IntEnum):
    """ index of state value for this bicycle model
    """
    X = 0
    Y = 1
    YAW = 2
    Vx = 3
    Vy = 4
    YAW_RATE = 5
    ACC_X = 6
    STEERING_ANGLE = 7

class ACTIONINDEX(IntEnum):
    """ index of action value for this bicycle model
    """
    ACC_X = 0
    STEERING_RATE = 1

class SingleTrackBicycle:
    def __init__(self, wheel_base, 
                 dt = 0.1, 
                 max_steering_angle: float = torch.pi / 3,
                 accel_time_constant: float = 0.2,
                 steering_angle_time_constant: float = 0.05,
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32} ):
        
        self.wheel_base = torch.tensor(wheel_base, **tensor_args)
        self.time_step = torch.tensor(dt, **tensor_args)
        self._max_steering_angle = max_steering_angle
        self._accel_time_constant = accel_time_constant
        self._steering_angle_time_constant = steering_angle_time_constant
        self.n_state =  len(STATEINDEX)
        self.n_action = len(ACTIONINDEX)
        self._tensor_args = tensor_args

    def __call__(self, state, action, t=None, dt=None):
        
        return self.forward(state, action, t, dt)

    def forward(self, state: torch.Tensor, action: torch.Tensor, t=None, dt =None):
        """ dynamic function of nuplan-like bicycle model for mppi

        Args:
            state : batch state [K x n_state] 
            action : control [K x n_action]
            t : current step [int] (not used)

        Returns:
            propagated state: [k x n_state]
        """
        # print(state.shape)
        dt = self.time_step if not dt else dt
        action = action.view(-1,self.n_action)
        state = state.view(-1,self.n_state)
        K, _ = state.shape
        
        state_X = state[:, STATEINDEX.X]
        state_Y = state[:, STATEINDEX.Y]
        state_Yaw = state[:, STATEINDEX.YAW]
        state_vx = state[:, STATEINDEX.Vx]
        state_vy = state[:, STATEINDEX.Vy]
        state_r = state[:, STATEINDEX.YAW_RATE]
        acc_x = state[:,STATEINDEX.ACC_X]
        steering_angle = state[:,STATEINDEX.STEERING_ANGLE]

        ideal_accel_x = action[:, ACTIONINDEX.ACC_X]
        ideal_steering_angle = action[:, ACTIONINDEX.STEERING_RATE] * dt + steering_angle

        next_acc_x = (
            dt
            / (dt + self._accel_time_constant)
            * (ideal_accel_x - acc_x)
            + acc_x
        )

        updated_steering_angle = (
            dt
            / (dt + self._steering_angle_time_constant)
            * (ideal_steering_angle - steering_angle)
            + steering_angle
        )
        next_steering_rate = (updated_steering_angle - steering_angle) / dt

        next_steering_angle = torch.clip(steering_angle + next_steering_rate * dt,
                                        min=-self._max_steering_angle,
                                        max=self._max_steering_angle)

        wheel_base = self.wheel_base

        

        next_X = state_X + state_vx * torch.cos(state_Yaw) * dt
        next_Y = state_Y + state_vx * torch.sin(state_Yaw) * dt
        next_Yaw = state_Yaw + (state_vx * torch.tan(next_steering_angle) / wheel_base) * dt
        next_vx = state_vx + next_acc_x * dt
        next_vy = state_vy * 0 # Lateral velocity is always zero in kinematic bicycle model
        next_r = next_vx * torch.tan(next_steering_angle) / wheel_base
        
        propagate_state = torch.zeros(size=(K, len(STATEINDEX)), **self._tensor_args)
        propagate_state[:,STATEINDEX.X] = next_X
        propagate_state[:,STATEINDEX.Y] = next_Y
        propagate_state[:,STATEINDEX.YAW] = next_Yaw
        propagate_state[:,STATEINDEX.Vx] = next_vx
        propagate_state[:,STATEINDEX.Vy] = next_vy
        propagate_state[:,STATEINDEX.YAW_RATE] = next_r
        propagate_state[:,STATEINDEX.ACC_X] = next_acc_x
        propagate_state[:,STATEINDEX.STEERING_ANGLE] = next_steering_angle

        return propagate_state
    
