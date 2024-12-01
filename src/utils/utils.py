from dataclasses import dataclass

@dataclass
class Bbox:
    """
    Represents a bounding box for a detected face with top-left and bottom-right corners.
    """
    x1: int
    y1: int
    x2: int
    y2: int

