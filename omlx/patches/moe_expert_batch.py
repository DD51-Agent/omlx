# SPDX-License-Identifier: Apache-2.0
"""Batched MoE offload install/evict path (A/B switch add-on).

A minimal, drop-in counterpart to the per-expert ``_install`` loop in
``moe_expert_offload.ExpertCache._ensure_ids``. This module keeps the
*upstream* ``moe_expert_offload.py`` byte-for-byte untouched and instead
installs an alternate ``_ensure_ids`` onto ``ExpertCache`` instances when
the batched path is enabled. The upstream install path remains the default
and is restored simply by not enabling the switch (or toggling ``cache.batch``
back to ``False``).

Why batch: a miss flood installs every miss with one ``_reserve`` + one
``_write`` per expert. The batch path collects all of a call's misses and
flushes them with a single atomic ``_reserve_many``/``_write_many`` pass,
which (a) reserves every goal slot in one consistent snapshot so no slot is
ever double-booked, and (b) reaches the steady-state eviction decision over
the whole batch instead of one greedy step at a time.

Correctness contract (identical to the upstream path):
* A miss never evicts an expert needed by the same call (``protected``).
* ``self.map`` stays an ``mx.int32`` MX array read by ``mx.take`` in
  ``_forward_overlap``/``_forward_expert_major``.
* ``slot_expert`` stays an ``np.int64`` array; the ``== -1`` sentinel and
  ``.pop(slot_of, ...)`` (never ``del``) conventions are preserved.
* Container types are preserved on every state mutation.

The pre-fix naive batched reservation (one ``_reserve`` per slot in a loop)
double-booked slots within a single call; do NOT reintroduce it. Keep the
single-snapshot atomic simulation below.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import wait

import mlx.core as mx
import numpy as np
from . import moe_expert_offload as _m

logger = logging.getLogger(__name__)

__all__ = ["batch", "apply_batch_install", "batch_ensure_ids"]

# Keep the whole upstream install/IO surface behind one module reference so
# this add-on stays a thin patch (no duplicated helpers).
_io_pool = _m._io_pool
_io_batch = _m._io_batch
_drain = _m._drain
_gpu_keepalive = _m._gpu_keepalive


def _reserve_many(cache, experts, protected=frozenset()) -> list:
    """Claim one slot per expert atomically; no slot is ever double-booked.

    The naive loop calling :meth:`_reserve` ``n`` times is broken: after
    step ``k`` evicts the lowest-score slot and marks it ``slot_expert=-1``,
    the next step's ``np.argmin`` re-selects that same slot (its score was
    not updated to inf), so two or more experts in the batch share a slot.
    We instead snapshot the current free/occupied sets, simulate all
    evictions in one pass, then apply the state changes atomically.

    Returns ``[(expert_id, slot), ...]`` in the order of ``experts``.
    """
    free_set = set(cache.free)
    current_slot_expert = dict(
        zip(range(len(cache.slot_expert)), cache.slot_expert.tolist())
    )
    current_slot_of = dict(cache.slot_of)
    current_map = dict(
        zip(range(len(cache.map)), cache.map.astype(mx.int64).tolist())
    )
    current_score = cache.score.copy()

    for e in protected:
        slot = current_slot_of.get(e)
        if slot is not None:
            current_score[slot] = np.inf

    new_slots: dict = {}
    free_stack = list(free_set)
    batch_slots: set = set()

    for e in experts:
        if e in current_slot_of:
            new_slots[e] = current_slot_of[e]
            continue
        if free_stack:
            slot = free_stack.pop()
        else:
            best_slot = None
            best_score = None
            for s, exp_id in current_slot_expert.items():
                if exp_id == -1 or s in batch_slots:
                    continue
                sc = current_score[s]
                if best_score is None or sc < best_score:
                    best_score = sc
                    best_slot = s
            slot = best_slot
            evicted_exp = current_slot_expert[slot]
            if evicted_exp != -1:
                del current_slot_of[evicted_exp]
                current_map[evicted_exp] = -1
                current_score[slot] = 0.0
                current_slot_expert[slot] = -1
        current_slot_expert[slot] = e
        current_slot_of[e] = slot
        current_map[e] = slot
        batch_slots.add(slot)
        new_slots[e] = slot

    cache.slot_expert = np.array(
        [current_slot_expert[s] for s in range(len(cache.slot_expert))],
        dtype=np.int64,
    )
    cache.slot_of = dict(current_slot_of)
    cache.map = mx.array(
        [current_map[e] for e in range(cache.n_experts)], dtype=mx.int32
    )
    cache.score = current_score
    cache.free = [
        s for s in range(len(cache.slot_expert)) if cache.slot_expert[s] == -1
    ]
    return [(e, new_slots[e]) for e in experts]


def _store(cache):
    """Resolve the staging/checkpoint store ops for ``cache`` (may be None)."""
    return (
        (cache._staging.to_mx, cache._staging.read)
        if cache._staging
        else (_m.CheckpointExpertStore.to_mx, _m.CheckpointExpertStore.read)
    )


def _write_many(cache, pairs, protected=frozenset()) -> None:
    """Install many experts in one call, with correct slot bookkeeping.

    ``pairs`` is ``[(expert_id, payload), ...]``. Calls ``_reserve_many``
    which simulates all evictions atomically so no slot is double-booked
    and protected experts are never evicted. Raises on partial write.
    """
    if not pairs:
        return
    expert_ids = [e for e, _ in pairs]
    assignments = _reserve_many(cache, expert_ids, protected)
    slot_of = dict(assignments)
    written: list = []
    convert, _read = _store(cache)
    # P1: batch the whole convert+install in one pass. Building every tensor
    # first and flushing them with a single ``mx.eval`` avoids forcing an MX
    # graph sync per field (the per-miss install loop syncs on every
    # ``.resident[...] =`` assignment). All tensor targets are recorded, then
    # one eval realises them all; assignment stays a plain object store.
    to_realise: list = []
    try:
        for e, payload in pairs:
            slot = slot_of[e]
            for name, field, plan, raw in payload:
                t = convert(plan, raw)
                cache.resident[name][field][slot] = t
                to_realise.append(t)
            written.append(e)
        if to_realise:
            mx.eval(*to_realise)
        cache.warm = len(cache.slot_of) == cache.n_experts
        cache.fetched_bytes += len(pairs) * cache.expert_bytes
    except BaseException:
        for e in written:
            cache.slot_of.pop(e, None)
        raise


def batch_ensure_ids(cache, ids, on_slow_read=None, pending=None) -> bool:
    """Batched replacement for ``ExpertCache._ensure_ids``.

    Structurally identical to the upstream method except that all of a
    call's misses are installed with one ``_write_many`` flush instead of
    one ``_install`` per expert. The read-ahead/prefetch overlap work is
    unchanged — only the slot install step is batched, which is the single
    variable we want to A/B against the upstream path.
    """
    cache._bind_staging()
    pending = dict(pending or {})
    needed = list(dict.fromkeys(int(e) for e in ids))
    if len(needed) > cache.capacity:
        _drain(pending)
        raise ValueError(
            "Expert cache capacity is smaller than the call's routes"
        )
    np.add.at(cache.score, np.asarray(ids, dtype=np.int64), 1.0)
    cache._calls += 1
    if cache._calls % _m._SCORE_DECAY_EVERY == 0:
        cache.score *= _m._SCORE_DECAY
    misses = [e for e in needed if e not in cache.slot_of]
    cache.hits += len(needed) - len(misses)
    if not misses:
        _drain(pending)
        return False
    protected = frozenset(needed)
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
        pending_pairs: list[tuple[int, list | None]] = []
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
                (
                    e,
                    [
                        (name, field, plan, f.result())
                        for name, field, plan, f in group
                    ],
                )
            )
            del pending[e], group
        if pending_pairs:
            _convert, read = _store(cache)
            _write_many(
                cache,
                [
                    (
                        e,
                        p
                        or [
                            (n, fld, pl, read(pl))
                            for n, fld, pl in cache._plans(e)
                        ],
                    )
                    for e, p in pending_pairs
                ],
                protected=protected,
            )
    finally:
        _drain(pending)
    return slow


def batch(cache) -> bool:
    """True when ``cache`` is running the batched install path."""
    return getattr(cache, "batch", False)


def apply_batch_install(cache, enabled: bool) -> None:
    """Toggle the batched install path on/off for one ``ExpertCache``.

    * ``enabled=True``: bind ``batch_ensure_ids`` as ``cache._ensure_ids``
      and set ``cache.batch=True``.
    * ``enabled=False``: restore the original (upstream) ``_ensure_ids`` and
      set ``cache.batch=False``.

    The pristine upstream method is captured on first call so it can always
    be restored — toggling off never loses the original install path.
    """
    import os as _os

    def _probe(msg: str) -> None:
        try:
            with open("/tmp/batch_probe_state.txt", "a") as _f:
                _f.write(msg + "\n")
        except Exception:
            pass

    if not hasattr(cache, "_batch_original_ensure_ids"):
        cache._batch_original_ensure_ids = type(cache)._ensure_ids
    if enabled:
        cache._ensure_ids = batch_ensure_ids.__get__(cache, type(cache))
        cache.batch = True
        _probe(f"ON  bound={batch_ensure_ids.__qualname__} cache.batch=True")
        logger.info("moe_expert_batch: batched install enabled for %s", cache)
    else:
        original = cache._batch_original_ensure_ids
        cache._ensure_ids = original.__get__(cache, type(cache))
        cache.batch = False
        _probe(f"OFF bound={getattr(original, '__qualname__', original)} cache.batch=False")
        logger.info("moe_expert_batch: upstream install restored for %s", cache)
