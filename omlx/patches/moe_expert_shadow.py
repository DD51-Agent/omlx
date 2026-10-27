# SPDX-License-Identifier: Apache-2.0
"""Shadow-tier upgrade for the MoE expert-offload slot space.

The upstream ``ExpertCache`` allocates a single tier of slots: every resident
expert competes for the same free list, so the hot working set churns — cold
misses evict warm experts, which are then re-fetched from the SSD.

This module upgrades one freshly constructed cache to a two-tier slot space
without touching the upstream class:

* slots ``[0, pcap)`` — the *primary* tier (hot core, sized by the resident
  fraction);
* slots ``[pcap, pcap + shadow)`` — the *shadow* tier (churn buffer).

Primary and shadow slots are scoring-equivalent: eviction is the whole-set
argmin over eligible residents, regardless of tier.  The tier split is
bookkeeping only; it decides where new slots land, not who may be evicted
first.  All behavior is opt-in: :func:`apply_shadow_upgrade` is called only
when ``moe_expert_offload_shadow_slots > 0``; with ``shadow = 0`` the cache
keeps the pristine upstream single-tier bookkeeping.

The upgrade binds per-instance methods (the same runtime-bind pattern as
:func:`omlx.patches.moe_expert_batch.apply_batch_install`):

* ``_allocate`` — the stock allocation, then re-partition the free lists;
* ``_reserve`` — two-tier claim (primary free, shadow free, then whole-set
  lowest-score eviction).

The batched install path (``moe_expert_batch``) reads the upgraded attributes
(``pcap`` / ``shadow`` / ``free_s``) through ``getattr`` with stock fallbacks,
so it composes with this upgrade.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["apply_shadow_upgrade", "resplit_free"]


def resplit_free(cache) -> None:
    """Re-partition the empty slots into the tiered free lists.

    ``cache.free`` holds the primary-tier empties, ``cache.free_s`` the
    shadow-tier empties.  The boundary is ``min(pcap, capacity)``: after a
    ``release_slots`` shrink the whole pool can live inside the primary tier,
    in which case ``free_s`` empties out.  Both lists mutate in place.
    """
    b = min(cache.pcap, cache.capacity)
    empt = np.where(cache.slot_expert < 0)[0]
    cache.free[:] = [int(i) for i in empt if i < b]
    cache.free_s[:] = [int(i) for i in empt if i >= b]


def _reserve_equivalent(self, protected=frozenset()) -> int:
    """Slot claim: primary free, then shadow free, then evict.

    Primary and shadow residents are scoring-equivalent, so eviction is the
    whole-set argmin over eligible residents.  The caller consumes the
    returned slot immediately (the stock ``_reserve`` contract); the batched
    install path re-appends freed slots via its own bulk eviction.
    """
    if self.free:
        return self.free.pop()
    if self.free_s:
        return self.free_s.pop()
    slots = self.slot_expert
    scores = self.score[slots].copy()
    scores[slots < 0] = np.inf
    idx = np.array(
        [int(self.slot_of[e]) for e in protected if e in self.slot_of],
        dtype=np.int64,
    )
    scores[idx] = np.inf
    occupied = slots >= 0
    if not np.any(occupied):
        raise RuntimeError("moe_expert_shadow: no occupied slot to evict")
    eligible = np.where(occupied & np.isfinite(scores))[0]
    if eligible.size == 0:
        raise RuntimeError("moe_expert_shadow: no eligible resident to evict")
    slot = int(eligible[int(np.argmin(scores[eligible]))])
    old_e = int(slots[slot])
    self.map[old_e] = -1
    slots[slot] = -1
    del self.slot_of[old_e]
    return slot


def _allocate_equivalent(self, capacity: int) -> None:
    """Stock reallocation, then re-partition the slot-index free lists."""
    type(self)._allocate(self, capacity)
    resplit_free(self)


def apply_shadow_upgrade(cache, primary: int, shadow: int) -> None:
    """Upgrade one freshly constructed ``ExpertCache`` to the two-tier space.

    The cache must have been constructed with
    ``capacity = min(primary + shadow, n_experts)`` slots (the offload hook
    does this); the upgrade only adds bookkeeping — ``pcap`` / ``shadow`` /
    ``free_s`` plus the tiered claim/allocation bindings.  ``primary`` is
    the configured resident count (before the shadow extension); when it
    clips against ``n_experts`` the shadow extends to nothing and the
    cache is left pristine.
    """
    pcap = min(primary, cache.n_experts)
    shadow_eff = min(max(0, int(shadow)), cache.n_experts - pcap)
    if not shadow_eff:
        return
    cache.pcap = pcap
    cache.shadow = shadow_eff
    cache.full_capacity = cache.pcap + cache.shadow
    cache.free_s = []
    resplit_free(cache)
    cache._reserve = _reserve_equivalent.__get__(cache, type(cache))
    cache._allocate = _allocate_equivalent.__get__(cache, type(cache))
    logger.info(
        "moe_expert_shadow: two-tier slots (primary=%d shadow=%d)",
        pcap,
        shadow_eff,
    )
