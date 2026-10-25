# SPDX-License-Identifier: Apache-2.0
"""Shadow-tier upgrade for the MoE expert-offload slot space.

The upstream ``ExpertCache`` allocates a single tier of slots: every resident
expert competes for the same free list, so the hot working set churns —
cold misses evict warm experts, which are then re-fetched from the SSD.

This module upgrades one freshly constructed cache to a two-tier slot space
without touching the upstream class:

* slots ``[0, pcap)`` — the *primary* tier (hot core, sized by the resident
  fraction);
* slots ``[pcap, pcap + shadow)`` — the *shadow* tier (churn buffer).

New/cold experts overflow into the shadow tier and eviction prefers it, so
the primary core keeps its working set.  All behavior is opt-in:
:func:`apply_shadow_upgrade` is called only when
``moe_expert_offload_shadow_slots > 0``; with ``shadow = 0`` the cache keeps
the pristine upstream single-tier bookkeeping.

The upgrade binds per-instance methods (the same runtime-bind pattern as
:func:`omlx.patches.moe_expert_batch.apply_batch_install`):

* ``_allocate`` — the stock allocation, then re-partition the free lists;
* ``_reserve`` — two-tier claim (primary free, shadow free, evict the
  lowest-scored shadow resident before touching the primary core).

The batched install path (``moe_expert_batch``) reads the upgraded
attributes (``pcap`` / ``shadow`` / ``free_s``) through ``getattr`` with
stock fallbacks, so it composes with this upgrade.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["apply_shadow_upgrade", "resplit_free"]


def resplit_free(cache) -> None:
    """Re-partition the empty slots into the tiered free lists.

    ``cache.free`` holds the primary-tier empties, ``cache.free_s`` the
    shadow-tier empties.  The tier boundary is ``min(pcap, capacity)``:
    after a ``release_slots`` shrink the whole pool can live inside the
    primary tier, in which case ``free_s`` simply empties out.  Both lists
    are mutated in place (the batched install path holds the same objects
    and rewrites them the same way).
    """
    b = min(cache.pcap, cache.capacity)
    empt = np.where(cache.slot_expert < 0)[0]
    cache.free[:] = [int(i) for i in empt if i < b]
    cache.free_s[:] = [int(i) for i in empt if i >= b]


def _reserve_tiered(self, protected=frozenset()) -> int:
    """Two-tier claim: primary free, then shadow free, then evict.

    Eviction prefers shadow-tier residents (the churn buffer) so the
    primary core keeps its working set; only when the shadow tier holds
    nothing evictable does it fall back to the whole-pool argmin.  The
    caller consumes the returned slot immediately (the stock ``_reserve``
    contract); the batched install path re-appends freed slots via its own
    bulk eviction.
    """
    if self.free:
        return self.free.pop()
    if self.free_s:
        return self.free_s.pop()
    slots = self.slot_expert
    scores = self.score[slots].copy()
    scores[slots < 0] = np.inf
    for e in protected:
        s = self.slot_of.get(e)
        if s is not None:
            scores[s] = np.inf
    b = min(self.pcap, self.capacity)
    if self.capacity > b:  # a shadow tier exists and is fully occupied
        seg = np.where(np.arange(self.capacity) >= b)[0]
        seg = seg[self.slot_expert[seg] >= 0]
        if seg.size:
            slot = int(seg[int(np.argmin(scores[seg]))])
            old_e = int(slots[slot])
            del self.slot_of[old_e]
            slots[slot] = -1
            self.map[old_e] = -1
            # No free-list append: the caller consumes the returned slot
            # immediately (legacy _reserve contract).  The next call
            # re-argmins; the batch path re-appends via _evict_bulk.
            return slot
    slot = int(np.argmin(scores))
    old_e = int(slots[slot])
    del self.slot_of[old_e]
    slots[slot] = -1
    self.map[old_e] = -1
    return slot


def _allocate_tiered(self, capacity: int) -> None:
    """Stock reallocation, then re-partition the tiered free lists."""
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
    cache._reserve = _reserve_tiered.__get__(cache, type(cache))
    cache._allocate = _allocate_tiered.__get__(cache, type(cache))
    logger.info(
        "moe_expert_shadow: two-tier slots (primary=%d shadow=%d)",
        pcap,
        shadow_eff,
    )
