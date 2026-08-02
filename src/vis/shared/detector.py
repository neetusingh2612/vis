"""Base interface every reflex/adaptive detector implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .contracts import Event, Message
from .state import VehicleState


class BaseDetector(ABC):
    """A detector inspects one message and optionally emits an Event.

    Reflex detectors MUST be O(1) per message and free of network/learning calls.
    """

    name: str = "base"

    @abstractmethod
    def inspect(self, msg: Message, state: VehicleState) -> Optional[Event]:
        ...

    def reset(self) -> None:
        """Clear any internal state (e.g. between dataset runs)."""
