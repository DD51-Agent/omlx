# SPDX-License-Identifier: Apache-2.0
"""MTP draft-window expert prefetch — couples the MTP and batch-expert lines.

Lightning MTP's draft phase (the MTP-head chain forwards) leaves every
offloaded target-layer expert pool idle on the IO side.  This module records
each layer's verify-routing history and, at the start of the draft window,
issues asynchronous read-ahead for the predicted next-verify expert set.
The next verify's ``ensure`` consumes the staged reads through the existing
``pending`` plumbing, so the SSD wait that used to serialize inside the
verify critical path now overlaps the draft compute.  MTP supplies the
window; the batch install line supplies the loader — neither line alone
produces the gain.

Enabled per model via ``moe_expert_mtp_prefetch`` (default off).  Only the
decode-scale path records: prefill and the expert-major chunk path never
reach the recording site, so the history holds one routing set per verify
cycle.  When the flag is off, caches carry no ``_mtp_pf`` state and every
hook here is a no-op — the off case is byte-identical to stock behavior.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, List, Optional

__all__ = [
    "install",
    "mark_verify",
    "end_verify",
    "begin_draft_window",
    "take_pending",
    "record",
    "is_verify",
]

_IDLE = "idle"
_VERIFY = "verify"
_DRAFT = "draft"


class _CachePF:
    """Per-cache prefetch state (one per offloaded MoE layer)."""

    __slots__ = ("hist", "limit", "pending", "phase")

    def __init__(self, window: int, limit: int, capacity: int):
        self.hist: Deque[frozenset] = deque(maxlen=max(1, window))
        self.limit = min(max(1, limit), capacity)
        self.pending: Optional[dict] = None
        self.phase = _IDLE


class _ModelKey:
    """Hashable wrapper for model objects that are not themselves hashable
    (MLX nn.Module subclasses dict; identity semantics via the wrapper's
    own id())."""

    __slots__ = ("_obj", "_id")

    def __init__(self, obj):
        self._obj = obj  # strong ref: keeps the key (and its id) stable
        self._id = id(obj)

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        return isinstance(other, _ModelKey) and self._id == other._id


# _ModelKey -> entry {"caches": [...]}; alias objects (language-model
# subobjects) share the entry of the object the mount loop installed on.
# Bounded by loaded models; strong refs keep loaded models alive anyway.
_entries: dict[_ModelKey, dict] = {}
_aliases: dict[_ModelKey, dict] = {}

# Instrumentation: were the hooks actually engaged?  Per-process counters,
# one log line per 8 staging calls.
_stages = 0
_staged_total = 0
_takes = 0
_taken_total = 0
_records = 0


def _entry(model: Any):
    if model is None:
        return None
    k = _ModelKey(model)
    return _entries.get(k) or _aliases.get(k)


def install(model: Any, cache: Any, window: int = 4, limit: int = 16) -> None:
    """Register one ``ExpertCache`` for draft-window prefetch (per layer).

    Idempotent: re-install refreshes the tuning knobs, never duplicates.
    """
    pf = getattr(cache, "_mtp_pf", None)
    if pf is None:
        pf = _CachePF(window, limit, cache.capacity)
        cache._mtp_pf = pf
    else:
        pf.limit = min(max(1, limit), cache.capacity)
    k = _ModelKey(model)
    entry = _entries.get(k)
    if entry is None:
        entry = {"caches": []}
        _entries[k] = entry
    if cache not in entry["caches"]:
        entry["caches"].append(cache)
    for alias in (getattr(model, "language_model", None),
                  getattr(model, "_language_model", None)):
        if alias is not None and alias is not model:
            ak = _ModelKey(alias)
            a = _aliases.get(ak)
            if a is None:
                a = {"caches": entry["caches"]}
                _aliases[ak] = a


def _set_phase(model: Any, phase: str) -> None:
    entry = _entry(model)
    if entry is None:
        return
    for c in entry["caches"]:
        c._mtp_pf.phase = phase


def mark_verify(model: Any) -> None:
    """Record routing while the target verify forward runs."""
    _set_phase(model, _VERIFY)


def end_verify(model: Any) -> None:
    """Stop recording once the verify forward has completed."""
    _set_phase(model, _IDLE)


def is_verify(cache: Any) -> bool:
    pf = getattr(cache, "_mtp_pf", None)
    return bool(pf and pf.phase == _VERIFY)


def record(cache: Any, ids: List[int]) -> None:
    """File this verify call's routing set into the layer's history."""
    global _records
    pf = getattr(cache, "_mtp_pf", None)
    if pf is not None and pf.phase == _VERIFY:
        pf.hist.append(frozenset(int(e) for e in ids if 0 <= int(e) < cache.n_experts))
        _records += 1
        if _records % 8 == 1:
            print(
                f"[mtp-pf] record #{_records} hist={len(pf.hist)} "
                f"uniq={len(frozenset().union(*pf.hist)) if pf.hist else 0}",
                flush=True,
            )


def _predict(pf: _CachePF, cache: Any) -> List[int]:
    """Predict the next verify's expert set from recency-weighted history.

    Recency order of ``hist`` (oldest first) means later cycles contribute
    more recent usage; a plain frequency rank over the window is a good
    proxy at the cycle cadence, and the pool's own eviction scores keep
    resident state authoritative for what is actually missing.
    """
    if not pf.hist:
        return []
    counts: dict[int, int] = {}
    for s in pf.hist:
        for e in s:
            counts[e] = counts.get(e, 0) + 1
    out: List[int] = []
    for e in sorted(counts, key=counts.get, reverse=True):
        if len(out) >= pf.limit:
            break
        if e not in cache.slot_of:
            out.append(e)
    return out


def begin_draft_window(model: Any) -> None:
    """Stage asynchronous read-ahead for the next verify, layer by layer.

    Runs on the host thread while the MTP head (GPU) drafts; the reads
    land on the idle target pools.  Staged reads are consumed by the next
    verify's ``take_pending``; a stale pending (a skipped cycle) is simply
    overwritten — its in-flight futures complete harmlessly.
    """
    entry = _entry(model)
    if entry is None:
        return
    global _stages, _staged_total
    _stages += 1
    _log = _stages % 8 == 1
    n_staged = 0
    for c in entry["caches"]:
        pf = c._mtp_pf
        pf.phase = _DRAFT
        predicted = _predict(pf, c)
        p = (c._read_ahead(predicted, pf.limit) or None) if predicted else None
        pf.pending = p
        n_staged += len(p) if p else 0
    _staged_total += n_staged
    if _log:
        print(
            f"[mtp-pf] stage #{_stages} cached_layers={len(entry['caches'])} "
            f"staged_this={n_staged} total_staged={_staged_total}",
            flush=True,
        )


def take_pending(cache: Any) -> Optional[dict]:
    """Pop and return this layer's staged read-ahead (or None)."""
    global _takes, _taken_total
    pf = getattr(cache, "_mtp_pf", None)
    if pf is None:
        return None
    p, pf.pending = pf.pending, None
    _takes += 1
    _taken_total += len(p) if p else 0
    if _takes % 8 == 1:
        print(
            f"[mtp-pf] take #{_takes} got={len(p) if p else 0} "
            f"total_got={_taken_total}",
            flush=True,
        )
    return p
