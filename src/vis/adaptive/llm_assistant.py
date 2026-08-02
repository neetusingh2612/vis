"""LLM assistant (Phase 2, OPTIONAL, ADVISORY ONLY).

May summarise decoy traces, draft a candidate signature, or explain an alert.
Its output has NO authority: it is just another candidate that must pass
negative selection + shadow mode + fleet validation. Keep it strictly off the
safety path. Even if manipulated, it cannot move the vehicle.

By default this runs fully offline and deterministic -- it drafts a candidate
from the incident itself, with no network call (so core stays dependency-light
and tests are hermetic). Inject a real model via `backend` (a callable
``(prompt: str) -> str``) behind the `.[ml]` extra; the backend only shapes the
human-readable rationale, never the structured target id, and the result still
goes through every downstream gate.
"""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .correlation_engine import Incident


class LLMAssistant:
    def __init__(self, backend: Optional[Callable[[str], str]] = None):
        self.backend = backend   # optional real LLM; advisory text only

    def suggest_candidate(self, incident: "Incident") -> dict:
        """Return a *draft* artifact dict; treated as UNPROVEN input downstream.

        The structured target (the arbitration id to match) is derived from the
        incident's own evidence, not invented by the model -- so a hallucinated
        or adversarial rationale cannot smuggle in a different target. Even so,
        the caller MUST still run this through negative selection + shadow mode.
        """
        aid = self._dominant_id(incident)
        if aid is None:
            # nothing to ground a rule on -> emit a non-runnable draft so it is
            # rejected by `matcher_for_artifact` rather than acted upon.
            return {"kind": None, "advisory": True,
                    "rationale": self._rationale(incident, None)}
        return {
            "kind": "id_match",
            "arbitration_id": aid,
            "advisory": True,                 # downstream must treat as unproven
            "source": "llm_assistant",
            "rationale": self._rationale(incident, aid),
        }

    def explain(self, incident: "Incident") -> str:
        """A short, human-readable account of an incident (advisory only)."""
        return self._rationale(incident, self._dominant_id(incident))

    def summarize_decoy_trace(self, events) -> str:
        """One-line summary of a decoy interaction trace (advisory only)."""
        surfaces = Counter(e.features.get("surface", "decoy") for e in events)
        parts = ", ".join(f"{n}x {s}" for s, n in surfaces.items()) or "no interactions"
        return f"decoy trace: {parts}"

    # -- internals --------------------------------------------------------- #
    def _rationale(self, incident: "Incident", aid: Optional[int]) -> str:
        target = f"arbitration id {aid:#x}" if isinstance(aid, int) else "an unclear target"
        prompt = (
            f"Summarise this vehicle-bus incident in one sentence. "
            f"class={incident.attack_class}, events={len(incident.events)}, "
            f"target={target}."
        )
        if self.backend is not None:
            return self.backend(prompt)
        return (f"{len(incident.events)} corroborating events implicate {target} "
                f"(suspected {incident.attack_class}).")

    @staticmethod
    def _dominant_id(incident: "Incident") -> Optional[int]:
        ids = [e.features.get("arbitration_id") for e in incident.events
               if isinstance(e.features.get("arbitration_id"), int)]
        if not ids:
            return None
        return Counter(ids).most_common(1)[0][0]
