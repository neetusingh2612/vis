"""Vehicle driving state -- consulted by the response engine so that no
containment action is ever unsafe for the current situation (design goal G8)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DriveMode(str, Enum):
    PARKED = "parked"
    LOW_SPEED = "low_speed"
    HIGHWAY = "highway"


@dataclass
class VehicleState:
    speed_kph: float = 0.0
    mode: DriveMode = DriveMode.PARKED

    @property
    def can_safe_stop(self) -> bool:
        """A full stop is only safe at low speed / when parked."""
        return self.mode in (DriveMode.PARKED, DriveMode.LOW_SPEED)
