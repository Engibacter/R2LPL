from enum import IntEnum

class RoadType(IntEnum):
    LANE = 1
    CONNECTOR = 2
    STOP_LINE = 3
    INTERSECTION = 4
    CROSSWALK = 5
    CARPARK = 6
    ROADBLOCK = 7
    ROADBLOCK_CONNECTOR = 8
    WALKWAYS = 9
    # DRIVABLE_AREA = 10
    INVALID = 0
    

    @classmethod
    def from_str(cls, label):
        label = label.upper()
        if label in cls.__members__:
            return cls[label]
        raise ValueError(f"Unknown road type: {label}")
