# SPDX-License-Identifier: Apache-2.0
"""LES-style lookahead expert prefetch for MoE expert offload.

Port of the core of llama.cpp-omni's Lookahead Expert Scheduling (LES) into
oMLX's MoE expert offload patch. Two-part design:

1. **Prediction (zero I/O)**: at layer L's forward, layer L+1's router gate
   is applied to the attention-post-norm hidden to predict which experts
   the next layer will route to. Prediction feeds the offload cache's
   eviction policy (protected set) so predicted experts are never evicted
   on the critical path.

2. **Prefetch (background I/O)**: the predicted experts' missing weights are
   submitted to the offload cache's existing IO thread pool while the GPU is
   busy with layer L's compute, so the next layer's miss reads are already
   in flight.

Numerical contract mirrors the offload patch itself: routing is unchanged —
the prediction only affects *when* expert weights are read and *which*
experts may be evicted, never *which* expert runs.

Multi-step extension:
   - ``arm_from_hidden`` now predicts a WINDOW of the next ``span`` MoE
     layers (default 8) instead of only the single next layer. Each layer in
     the window uses its own registered router gate on the SAME hidden, so
     the cost is ``span`` small matmuls instead of 1; deduplication across
     the window is done once, per layer.
   - **Drop-when-busy**: before submitting reads, the predictor probes
     whether the previous prefetch batch is still in flight via a
     ``cache.prefetch_in_flight()`` accessor (omni's ``n_arm_skipped``
     equivalent). When the batch is still running the submission is SKIPPED,
     not queued: queuing only delays later layers' critical-path reads.

"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx

# Offload cache accessors. The predictor is engine-agnostic: it holds no
# reference to a cache, the offload patch passes one in.
# _resident_sets[layer_idx] -> set of resident expert ids (for hit counting)


class ExpertLookaheadPredictor:
    """Predicts next-layer expert routing for offload prefetch/eviction.

    ``gate_w_for(layer_idx)`` returns the router gate weight ``[E, hidden]``
    for the layer with the given model-layer index. Weights come from the
    model's resident parameters, not disk.

    ``span``: number of successive MoE layers (starting at ``layer_idx+1``)
    to predict in one arm event. 1 = legacy single-step; 8 = default.
    """

    def __init__(
        self,
        gate_w_for: Callable[[int], mx.array],
        depth: int = 8,
        read_n: int = -1,
        span: int = 8,
    ):
        self._gate_w_for = gate_w_for
        self.depth = max(0, min(int(depth), 16))
        self.read_n = int(read_n) if int(read_n) >= 0 else self.depth
        self.span = max(1, min(int(span), 16))
        self._states: dict[int, dict] = {}
        # Diagnostics, aligned with omni's counters.
        self.n_armed = 0
        self.n_cached = 0
        self.n_read_submitted = 0
        self.n_skipped_busy = 0
        self._resident_sets: dict[int, set] = {}

    # -- registration -------------------------------------------------------

    def register(self, layer_idx: int, gate_w: mx.array) -> None:
        """Register the router gate weight ``[E, hidden]`` of one MoE layer."""
        self._states[layer_idx] = {"gate_w": gate_w, "last_pred": []}

    # -- prediction ---------------------------------------------------------

    def predict(self, layer_idx: int, hidden: mx.array) -> list[int]:
        """Top-k expert ids for ``layer_idx`` given the hidden state that
        layer will consume. Returns [] when prediction is disabled.

        ``hidden``: single row ``[hidden]`` or batched ``[b, hidden]``;
        decode (b==1) is the supported case.
        """
        if self.depth <= 0 or layer_idx not in self._states:
            return []
        st = self._states[layer_idx]
        try:
            gate_w = st["gate_w"]
            # Normalize to [b, hidden] regardless of input rank: decode x can
            # arrive as [hidden], [1, hidden] or [1, s, hidden].
            x = hidden.reshape((-1, hidden.shape[-1]))     # [b, hidden]
            logits = mx.matmul(x, gate_w.T)                 # [b, E]
            k = min(self.depth, gate_w.shape[0])
            # argsort ascending; the largest k expert ids are the last k
            # columns in descending order.
            order = mx.argsort(logits, axis=-1)            # [b, E]
            topk_ids = order[:, ::-1][:, :k].tolist()      # [[...k ints] x b]
            pred = []
            for row in topk_ids:
                if isinstance(row, (int, float)):
                    pred.append(int(row))
                else:
                    pred.extend(int(v) for v in row)
            st["last_pred"] = pred
            self._note_cached(layer_idx, pred)
            return pred
        except Exception:
            # Prediction must never break the decode critical path: any
            # shape/dtype surprise degrades to "no prefetch this step".
            return []

    def _predict_window(
        self, cur_idx: int, hidden: mx.array
    ) -> list[tuple[int, list[int]]]:
        """Predict top-k for the next ``span`` MoE layers (all consuming the
        same hidden). Returns ``[(layer_idx, pred), ...]`` for layers in
        ``[cur_idx+1, cur_idx+span]`` that have a registered gate.

        Each entry is the layer's own top-k (independent, not deduplicated
        across the window): the cache dedupes per-layer on its own;
        cross-layer dedup would only skip reads that the target layer's
        own ``_ensure_ids`` would have to issue anyway when it runs.
        """
        out: list[tuple[int, list[int]]] = []
        for j in range(1, self.span + 1):
            tgt = cur_idx + j
            if tgt in self._states:
                pred = self.predict(tgt, hidden)
                if pred:
                    out.append((tgt, pred))
        return out

    def _note_cached(self, layer_idx: int, pred: list[int]) -> None:
        resident = self._resident_sets.get(layer_idx, set())
        self.n_cached += sum(1 for e in pred if e in resident)

    def set_resident_set(self, layer_idx: int, resident: set[int]) -> None:
        """Update the resident-expert set for a layer (hit accounting)."""
        self._resident_sets[layer_idx] = set(resident)

    @property
    def stats(self) -> dict:
        return {
            "depth": self.depth,
            "read_n": self.read_n,
            "span": self.span,
            "n_armed": self.n_armed,
            "n_cached": self.n_cached,
            "n_read_submitted": self.n_read_submitted,
            "n_skipped_busy": self.n_skipped_busy,
        }

    # -- prefetch submission ------------------------------------------------

    def arm_from_hidden(self, cache, layer_idx: int, hidden: "mx.array") -> None:
        """Predict from ``hidden`` then arm prefetch in one step.

        This is the call site hook: the offload patch passes the layer's
        output hidden state; the predictor computes the next-``span`` layers'
        top-k and submits background reads for any missing predicted experts.

        Drop-when-busy: if the cache reports its previous prefetch
        batch is still in flight, skip this submission entirely instead of
        queueing it. Counters exposed via
        :attr:`stats`.
        """
        if self.read_n <= 0 or self.depth <= 0:
            return
        if not hasattr(cache, "_read_ahead"):
            return
        # Drop-when-busy probe (best effort; older caches lack it).
        probe = getattr(cache, "prefetch_in_flight", None)
        if probe is not None:
            try:
                if probe():
                    self.n_skipped_busy += 1
                    return
            except Exception:
                pass

        if self.span <= 1:
            # Legacy single-step path, unchanged behavior.
            pred = self.predict(layer_idx, hidden)
            if pred:
                tgt = layer_idx + 1
                actual = getattr(cache, "last_actual_routes", {}).get(tgt)
                if actual is not None:
                    pred_set = set(int(v) for v in pred)
                    self.pred_count += len(pred_set)
                    self.pred_hits += len(pred_set & actual)
                self.arm_prefetch(cache, tgt, pred)
            return

        windows = self._predict_window(layer_idx, hidden)
        for tgt, pred in windows:
            # Effectiveness: compare predicted ids against the target layer's
            # actual route set (recorded by the offload patch when that layer
            # last ran; None the first time).
            actual = getattr(cache, "last_actual_routes", {}).get(tgt)
            if actual is not None:
                pred_set = set(int(v) for v in pred)
                self.pred_count += len(pred_set)
                self.pred_hits += len(pred_set & actual)
            self.arm_prefetch(cache, tgt, pred)

    def arm_prefill_warmup(self, cache, layer_idx: int, hidden: "mx.array") -> None:
        """Single-step (span=1) prediction at the prefill/decode
        boundary.

        Called once at the end of an expert-major prefill call, with the
        prefill's last token hidden as the decode step 1 input for this layer.
        Only the immediate next layer's top-k matters here (later layers arm
        their own predictions on their own decode steps), so this bypasses
        the multi-step ``span`` window on purpose.
        """
        if self.depth <= 0 or not pred_ok(self.depth):
            return
        if not hasattr(cache, "_read_ahead"):
            return
        pred = self.predict(layer_idx, hidden)
        if pred:
            tgt = layer_idx + 1
            actual = getattr(cache, "last_actual_routes", {}).get(tgt)
            if actual is not None:
                pred_set = set(int(v) for v in pred)
                self.pred_count += len(pred_set)
                self.pred_hits += len(pred_set & actual)
            self.arm_prefetch(cache, tgt, pred)

    def arm_prefetch(self, cache, layer_idx: int, pred: list[int]) -> None:
        """Submit background reads for the predicted-but-missing experts via
        the offload cache's IO thread pool (uses cache._read_ahead)."""
        if self.read_n <= 0 or not pred:
            return
        if not hasattr(cache, "_read_ahead"):
            return
        # _read_ahead already dedupes and limits to read_n; it also checks
        # slot_of internally. The cache handles everything.
        missing = [e for e in pred[: self.read_n] if e not in cache.slot_of]
        if not missing:
            return
        pending = cache._read_ahead(missing, self.read_n)
        self.n_read_submitted += len(pending)
        self.n_armed += 1


def pred_ok(depth: int) -> bool:
    """depth is usable for prediction (positive and within the registered cap)."""
    return 0 < depth <= 16





__all__ = [
    "ExpertLookaheadPredictor",
    "pred_ok",
]
