"""``NewTokenRatioTracker`` — owns the scheduler's KV-budget headroom factor.

``new_token_ratio`` is an estimate (in ``[min, init]``) of how many extra
KV tokens each running request will consume before it finishes. It seeds
``PrefillAdder``'s admission policy:

- Starts at ``init`` (computed from ``SGLANG_INIT_NEW_TOKEN_RATIO`` and
  ``server_args.schedule_conservativeness``).
- Decays toward ``min`` each non-retract decode step (subtracting
  ``decay``, clamped to ``min``) — confidence grows as the batch stays
  feasible.
- On a forced retract (KV-cache pool full), jumps back up to the
  post-retract estimate produced by
  ``ScheduleBatch._estimate_new_token_ratio_after_retract``.
- On scheduler idle, resets to ``init``.

Packaging the four sibling attributes (``init``, ``min``, ``decay``,
``current``) into one tracker collapses the cluster on ``Scheduler``
and turns the three state transitions into named methods.
"""

from dataclasses import dataclass

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs


@dataclass(slots=True, kw_only=True)
class NewTokenRatioTracker:
    init: float
    min: float
    decay: float
    current: float

    @classmethod
    def from_server_args(cls, server_args: ServerArgs) -> "NewTokenRatioTracker":
        init = min(
            envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()
            * server_args.schedule_conservativeness,
            1.0,
        )
        min_ratio = min(
            init * envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get(),
            1.0,
        )
        decay = (init - min_ratio) / envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get()
        return cls(init=init, min=min_ratio, decay=decay, current=init)

    def decay_step(self) -> None:
        """Decay ``current`` by one step toward ``min``."""
        self.current = max(self.current - self.decay, self.min)

    def reset(self) -> None:
        """Reset ``current`` back to ``init`` (used on scheduler idle)."""
        self.current = self.init
