"""Negative selection (Phase 2).

The on-vehicle false-positive gate (immune-inspired). A candidate detector is
tested against a stored 'self' corpus of known-good traffic and DISCARDED if it
would fire on normal behaviour -- the system learns *self-tolerance* before it
is ever allowed to react. Capture the self corpus BEFORE running this.

A "candidate" here is a *matcher*: a callable ``(Message) -> bool`` that returns
True on the traffic it intends to flag (see ``antibody_generator``).
"""
from __future__ import annotations

from typing import Callable, Iterable

from ..shared.contracts import Message

Matcher = Callable[[Message], bool]


class NegativeSelection:
    def __init__(self, self_corpus: Iterable[Message] | None = None):
        # capture BEFORE running this (Phase 2 step 1)
        self.self_corpus: list[Message] = list(self_corpus or [])

    def add_self(self, messages: Iterable[Message]) -> None:
        self.self_corpus.extend(messages)

    def passes(self, candidate: Matcher) -> bool:
        """True if `candidate` does NOT fire on any known-good sample."""
        return not any(candidate(m) for m in self.self_corpus)
