from dataclasses import dataclass

@dataclass
class Waypoint:
    lat: float = 0
    lon: float = 0
    alt: float = 0
    reach: int = 0