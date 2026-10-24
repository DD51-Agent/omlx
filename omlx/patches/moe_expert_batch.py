# SPDX-License-Identifier: Apache-2.0
"""Batched install add-on for the MoE expert offload path.

Thin wrapper: read/evict logic is unchanged from upstream; only the
install step is simplified to one per-expert write pass over a
``cache.free`` list that was topped up up-front by a single bulk
eviction.  No validation loops, no per-expert argmin.
"""

from __future__ import annotations

import logging
import os
import time

from concurrent.futures import wait
import mlx.core as mx
import numpy as np
from . import moe_expert_offload as _m

logger = logging.getLogger(__name__)

__all__ = ["apply_batch_install", "batch_ensure_ids"]

_io_pool = _m._io_pool
_io_batch = _m._io_batch
_drain = _m._drain
_gpu_keepalive = _m._gpu_keepalive


def _sync_mode() -> int:
    """Batch-install sync level (env OMLX_MOE_BATCH_SYNC).

    0  legacy pre-10-22: no flush (in-flight draft/verify reads may tear on
       a recycled slot — do not use with MTP).
    1  default: pre-evict flush only.  The single mx.eval() before occupied
       slots are freed guarantees in-flight gathers have read their old
       values; under single-stream in-order execution the slot writes queue
       ahead of the caller's next dispatches, so per-write evals are
       redundant.
    2  10-22 legacy: pre-evict flush + per-write evals (N full pipeline
       drains per burst — measured ~35% hidden wall tax at sparse-miss
       residency, it was eating the MTP gain at r0.75).  Kept only as a
       determinism fallback.
    """
    try:
        return int(os.environ.get("OMLX_MOE_BATCH_SYNC", "1"))
    except ValueError:
        return 1


def _evict_bulk(cache, need: int, protected=frozenset()) -> None:
    """Ensure at least ``need`` slots are in ``cache.free`` (one bulk pass).

    Empty slots are always safe (unreferenced, no one to evict); only the
    shortfall is chosen from occupied slots, as the ``need`` lowest-scored
    residents with the call's experts protected.  Existing free slots are
    never consumed here.
    """
    if need <= 0:
        return
    slots = cache.slot_expert
    empty = slots < 0
    n_empty = int(np.count_nonzero(empty))
    need -= n_empty
    if need <= 0:
        for i in np.where(empty)[0]:
            cache.free.append(int(i))
        return
    occupied = np.where(~empty)[0]
    if occupied.size < need:
        raise RuntimeError(
            f"moe_expert_batch: cannot free {n_empty + need} slots "
            f"(occupied={occupied.size})"
        )
    scores = cache.score[slots].copy()
    scores[empty] = np.inf
    for e in protected:
        s = cache.slot_of.get(e)
        if s is not None:
            scores[s] = np.inf
    b = min(getattr(cache, "pcap", cache.capacity), cache.capacity)
    if getattr(cache, "shadow", 0) > 0 and cache.capacity > b:
        # Tiered eviction: shadow-tier residents (the churn buffer) are
        # freed before primary-tier residents, so the primary core keeps
        # its working set.  Within each tier, lowest score first; the tier
        # boundary dominates (no cross-tier interleaving).  When the shadow
        # tier was released (capacity collapsed to the floor) this degrades
        # to the legacy whole-pool ordering.
        shadow_occ = occupied[occupied >= b]
        primary_occ = occupied[occupied < b]
        s_order = shadow_occ[np.argsort(scores[shadow_occ], kind="stable")]
        p_order = primary_occ[np.argsort(scores[primary_occ], kind="stable")]
        evict_ids = np.concatenate([s_order, p_order])
    else:
        evict_ids = occupied
    for v in evict_ids[:need]:
        slot = int(v)
        oe = int(slots[slot])
        del cache.slot_of[oe]
        cache.map[oe] = -1
        slots[slot] = -1
        (cache.free_s if slot >= b else cache.free).append(slot)
    if _sync_mode() >= 1:
        # A just-freed slot may still be read by in-flight kernels (the MTP
        # pipeline queues draft/verify gathers without per-step evals).
        # Overwriting it before those reads complete yields torn data — the
        # batch x MTP nondeterminism.  Flush before _write_many recycles it.
        # Single flush: post-write evals are redundant under single-stream
        # in-order execution (mode 2 restores the legacy per-write evals).
        mx.eval()


def _store(cache):
    return (
        (cache._staging.to_mx, cache._staging.read)
        if cache._staging
        else (_m.CheckpointExpertStore.to_mx, _m.CheckpointExpertStore.read)
    )


def _write_many(cache, pairs, protected=frozenset()) -> None:
    """Install all pairs: top up cache.free up-front, then one per-expert write pass.

    Pop from the free list (which may include slots freed by prior bulk
    evictions or by upstream's own per-expert installs) and evict only the
    shortfall.  No validation loops, no per-expert argmin.
    """
    if not pairs:
        return
    convert, read = _store(cache)
    # Authoritative empty-slot count from slot_expert (the single source of
    # truth).  cache.free may hold stale/duplicate entries: upstream
    # _reserve never appends evicted slots back, but _evict_bulk does and
    # install failures append — len(cache.free) can overcount the true
    # free set, which starves the eviction below and lets the pop loop
    # hand one slot to two experts (shared-slot corruption -> repeated
    # output).
    slots = cache.slot_expert
    _misses = [e for e, _ in pairs if e not in cache.slot_of]
    # _evict_bulk's contract: ``need`` is the TARGET free count (it subtracts
    # the current empty set internally, then evicts the shortfall).  Pass
    # the full miss count — an earlier version passed ``len(_misses)-n_free``
    # which double-subtracted the empty set and under-evicted; the
    # over-appended free list masked that as shared-slot corruption.
    _evict_bulk(cache, len(_misses), protected)
    # Rebuild the tiered free lists from the authoritative source so the pop
    # loop below can never see duplicate/stale entries.  Evicted slots are
    # already flushed by _evict_bulk's pre-evict mx.eval() (sync mode >=1)
    # when OMLX_MOE_BATCH_SYNC=1; in-order execution makes the per-write
    # evals redundant here.  Allocation prefers the primary tier (hot core),
    # overflow goes to the shadow tier.
    b = min(getattr(cache, "pcap", cache.capacity), cache.capacity)
    empt = [int(i) for i in np.where(slots < 0)[0]]
    cache.free[:] = [i for i in empt if i < b]
    if hasattr(cache, "free_s"):
        cache.free_s[:] = [i for i in empt if i >= b]
    for e, payload in pairs:
        if e in cache.slot_of:
            continue
        if cache.free:
            slot = cache.free.pop()
        elif getattr(cache, "free_s", None):
            slot = cache.free_s.pop()
        else:
            # _evict_bulk guaranteed enough empties; reaching here means a
            # bookkeeping regression (shared-slot corruption) — fail loud.
            raise RuntimeError("moe_expert_batch: free-slot shortfall after evict")
        cache.slot_expert[slot] = e
        cache.map[e] = slot
        cache.slot_of[e] = slot
        written = set()
        for name, field, plan, raw in payload:
            cache.resident[name][field][slot] = convert(plan, raw)
            written.add((name, field))
        if _sync_mode() >= 2:
            # 10-22 legacy per-write eval (OMLX_MOE_BATCH_SYNC=2 fallback):
            # N full pipeline drains per burst — measured ~35% hidden wall tax
            # at sparse-miss residency (r0.75).  Redundant under single-stream
            # in-order execution; the pre-evict flush (mode>=1) is the only
            # sync the torn-read invariant actually requires.
            mx.eval(*[cache.resident[n][f] for n, f in written])
    cache.warm = len(cache.slot_of) == cache.n_experts
    cache.fetched_bytes += len(pairs) * cache.expert_bytes


def batch_ensure_ids(cache, ids, on_slow_read=None, pending=None, protect=None) -> bool:
    """:meth:`_ensure_ids` with the install step replaced by :func:`_write_many`."""
    cache._bind_staging()
    pending = dict(pending or {})
    needed = list(dict.fromkeys(int(e) for e in ids if 0 <= int(e) < cache.n_experts))
    if len(needed) > cache.capacity:
        _drain(pending)
        raise ValueError("Expert cache capacity is smaller than the call's routes")
    np.add.at(cache.score, np.asarray(needed, dtype=np.int64), 1.0)
    cache._calls += 1
    if cache._calls % _m._SCORE_DECAY_EVERY == 0:
        cache.score *= _m._SCORE_DECAY
    misses = [e for e in needed if e not in cache.slot_of]
    cache.hits += len(needed) - len(misses)
    if not misses:
        _drain(pending)
        return False
    protected = frozenset(needed) | frozenset(protect or ())
    pool = _io_pool()
    window = _io_batch()
    sent = 0
    keepalive = barrier = None
    slow = False

    def prefetch(upto: int) -> None:
        nonlocal sent
        if pool is None:
            return
        while sent < min(upto, len(misses)):
            e = misses[sent]
            sent += 1
            if e not in pending:
                pending[e] = cache._submit(pool, e)

    try:
        pending_pairs = []
        prefetch(window)
        deadline = time.perf_counter() + _m._OVERLAP_GRACE_S
        for done, e in enumerate(misses):
            cache.misses += 1
            prefetch(done + window)
            group = pending.get(e)
            if on_slow_read is not None:
                futures = [f for _, _, _, f in group or ()]
                timeout = max(0.0, deadline - time.perf_counter())
                if futures and wait(futures, timeout=timeout).not_done:
                    slow = True
                    barrier = on_slow_read()
                    keepalive = _gpu_keepalive()
                on_slow_read = None
            if group is None:
                pending_pairs.append((e, None))
                continue
            if keepalive is not None:
                keepalive.wait([f for _, _, _, f in group])
            if barrier is not None:
                mx.eval(barrier)
                barrier = None
            pending_pairs.append(
                (e, [(n, fld, pl, f.result()) for n, fld, pl, f in group]),
            )
            del pending[e], group
        if pending_pairs:
            convert, read = _store(cache)
            _write_many(
                cache,
                [(e, p or [(n, fld, pl, read(pl)) for n, fld, pl in cache._plans(e)])
                 for e, p in pending_pairs],
                protected=protected,
            )
    finally:
        _drain(pending)
    return slow


def apply_batch_install(cache) -> None:
    """Bind the batched install path onto one ``ExpertCache``.

    Called only when the settings flag is on; the off case keeps the
    upstream native ``_ensure_ids`` untouched (never invoked here).
    """
    if not hasattr(cache, "_batch_original_ensure_ids"):
        cache._batch_original_ensure_ids = type(cache)._ensure_ids
    cache._ensure_ids = batch_ensure_ids.__get__(cache, type(cache))
    cache.batch = True
    logger.info("moe_expert_batch: batched install enabled for %s", cache)
