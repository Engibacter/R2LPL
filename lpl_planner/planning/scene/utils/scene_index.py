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