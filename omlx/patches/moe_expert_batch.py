# SPDX-License-Identifier: Apache-2.0
"""Batched install/evict add-on for the MoE expert offload path. """

from __future__ import annotations

import logging
import time
from concurrent.futures import wait

import mlx.core as mx
import numpy as np
from . import moe_expert_offload as _m

logger = logging.getLogger(__name__)

__all__ = ["batch", "apply_batch_install", "batch_ensure_ids"]

# Upstream install/IO surface, reused so this add-on stays a thin patch.
_io_pool = _m._io_pool
_io_batch = _m._io_batch
_drain = _m._drain
_gpu_keepalive = _m._gpu_keepalive


def _reserve_many(cache, experts, protected=frozenset()) -> list:
    """Reserve one slot per expert atomically in a single simulation pass."""
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
            best_slot, best_score = None, None
            for s, exp_id in current_slot_expert.items():
                if exp_id == -1 or s in batch_slots:
                    continue
                sc = current_score[s]
                if best_score is None or sc < best_score:
                    best_score, best_slot = sc, s
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

    for e2 in list(current_slot_of.keys()):
        if e2 not in new_slots:
            del current_slot_of[e2]
    for e, slot in new_slots.items():
        cache.map[e] = slot
        cache.slot_expert[slot] = e
        cache.slot_of[e] = slot
    cache.score = current_score
    cache.free = [
        s for s in range(len(cache.slot_expert)) if cache.slot_expert[s] == -1
    ]
    return [(e, new_slots[e]) for e in experts]


def _store(cache):
    """Resolve staging/checkpoint store ops for ``cache`` (may be None)."""
    return (
        (cache._staging.to_mx, cache._staging.read)
        if cache._staging
        else (_m.CheckpointExpertStore.to_mx, _m.CheckpointExpertStore.read)
    )


def _write_many(
    cache, pairs, protected=frozenset(), chunk: int | None = None
) -> None:
    """Install many experts in batches of at most ``chunk``; sequential.

    The install is split into batches of at most ``chunk`` experts (``chunk``
    None/<=0 -> one batch). Each batch is *fully independent*: slot
    reservation runs on the current cache state, then convert + write
    complete the batch before the next one starts. Later batches therefore
    observe earlier batches' evictions — a 7-miss + chunk=2 schedule is
    2,2,2,1 (4 batches). When ``chunk`` is None the whole install is a
    single batch.

    No mx.eval is issued at batch boundaries: ``_reserve_many`` commits
    slot assignments as scalar index writes (same shape as upstream's
    per-expert install loop), so Metal's single-stream ordering makes them
    visible to the downstream ``mx.take`` in qmm without an explicit sync
    point — matching upstream's measured-zero-eval install behaviour.
    Weight payloads written into ``cache.resident`` stay in the Metal
    lazy queue and are realised once by the prefill chunk's own eval,
    exactly like upstream.

    Raises on partial write; already-installed batches stay in place."""
    if not pairs:
        return
    convert, _read = _store(cache)
    step = chunk if chunk and chunk > 0 else len(pairs)
    try:
        for start in range(0, len(pairs), step):
            batch = pairs[start : start + step]
            assignments = _reserve_many(
                cache, [e for e, _ in batch], protected
            )
            slot_of = dict(assignments)
            for e, payload in batch:
                slot = slot_of[e]
                for name, field, plan, raw in payload:
                    t = convert(plan, raw)
                    cache.resident[name][field][slot] = t
        cache.warm = len(cache.slot_of) == cache.n_experts
        cache.fetched_bytes += len(pairs) * cache.expert_bytes
    except BaseException:
        raise


def batch_ensure_ids(cache, ids, on_slow_read=None, pending=None) -> bool:
    """Batched replacement for ``ExpertCache._ensure_ids``. Only the miss
    install step is batched; read-ahead/prefetch overlap is unchanged.

    The install is split into sequential batches of at most
    ``moe_expert_offload_batch_chunk`` experts (clamped to [2, io_window],
    default 4): each batch reserves slots on the current cache state, then
    converts, writes, then the next batch starts — no explicit flush
    between batches; downstream prefill's per-chunk eval materialises
    the pending weight writes, exactly like upstream. The chunk is read
    once per call from ``cache.model_settings`` so it can change between
    model reloads without a server restart."""
    cache._bind_staging()
    settings = getattr(cache, "model_settings", None)
    raw_chunk = getattr(settings, "moe_expert_offload_batch_chunk", 4)
    # Clamp to [2, io_window]: below 2 the prefetch window makes chunking
    # pointless, and above the in-flight read window the chunk would only
    # hold payloads whose reads already finished, i.e. it would grow memory
    # bandwidth pressure for no overlap benefit.
    chunk = min(max(int(raw_chunk or 4), 2), max(_io_batch(), 2))
    pending = dict(pending or {})
    needed = list(dict.fromkeys(int(e) for e in ids))
    if len(needed) > cache.capacity:
        _drain(pending)
        raise ValueError("Expert cache capacity is smaller than the call's routes")
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
                ),
            )
            del pending[e], group
        if pending_pairs:
            _convert, read = _store(cache)
            logger.info(
                "moe_expert_batch: installing %d expert(s) this batch",
                len(pending_pairs),
            )
            _write_many(
                cache,
                [
                    (
                        e,
                        p
                        or [
                            (n, fld, pl, read(pl)) for n, fld, pl in cache._plans(e)
                        ],
                    )
                    for e, p in pending_pairs
                ],
                protected=protected,
                chunk=chunk,
            )
    finally:
        _drain(pending)
    return slow


def batch(cache) -> bool:
    """True when ``cache`` is running the batched install path."""
    return bool(getattr(cache, "batch", False))


def apply_batch_install(cache, enabled: bool) -> None:
    """Toggle the batched install path on/off for one ``ExpertCache``."""
    # Capture the pristine upstream method once so it can always be restored.
    if not hasattr(cache, "_batch_original_ensure_ids"):
        cache._batch_original_ensure_ids = type(cache)._ensure_ids
    if enabled:
        cache._ensure_ids = batch_ensure_ids.__get__(cache, type(cache))
        cache.batch = True
        logger.info("moe_expert_batch: batched install enabled for %s", cache)
    else:
        original = cache._batch_original_ensure_ids
        cache._ensure_ids = original.__get__(cache, type(cache))
        cache.batch = False
        logger.info("moe_expert_batch: upstream install restored for %s", cache)
