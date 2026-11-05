"""CUDA graphs for the decoder of single-request image and video inputs.

Enable with ``LoadOptions(media_cuda_graphs=True)``, ``JEVANY_MEDIA_CUDA_GRAPHS=1`` or ``--media-cuda-graphs``.

``RowGraphs`` (``jevany.cudagraphs``) replays text rows; image and video requests otherwise run the whole backbone
eagerly. Here the vision tower and the M-RoPE position index stay eager, and the language decoder, whose many small
kernels dominate a launch-bound batch-1 forward, is captured once per padded length and replayed. ``inputs_embeds``
and the position ids are copied into static buffers and the tail is right-padded with zero embeddings and continuing
positions; every decoder layer is causal, so padding never reaches a real token. Captures are lazy (the first request
of a length pays for it) and share one memory pool. Capture makes the eager attention-mask decisions of an unpadded
single row (``_eager_mask_decisions``), so SDPA keeps its ``is_causal`` FlashAttention path inside the graphs.

Everything else runs eagerly: text rows, batches, padding masks, gradients, caches, positions that do not rise by one,
calls during another graph capture and requests longer than ``max_tokens``. Replays reuse static buffers, so calls
must not overlap; ``DecisionRuntime`` serializes inference. Probabilities are close to, not bit-identical with, the
eager path, because padded shapes select different kernels.
"""
from __future__ import annotations

import time

import torch

# Padded lengths; the spacing bounds padding at 25-50% of the request.
LENGTHS = (128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048,
           2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192)
_FORWARD_KEYS = frozenset({"input_ids", "inputs_embeds", "position_ids", "attention_mask", "past_key_values",
                           "use_cache", "return_dict"})


class MediaGraphs:
    """Replacement ``forward`` for a Transformers decoder; ``stats`` reports lengths, capture time and call counts."""

    def __init__(self, decoder: torch.nn.Module, max_tokens: int = 2048):
        if type(max_tokens) is not int or max_tokens < LENGTHS[0]:
            raise ValueError(f"media_cuda_graph_max_tokens must be an integer of at least {LENGTHS[0]}")
        self.decoder, self.eager = decoder, decoder.forward
        self.lengths = tuple(n for n in LENGTHS if n <= max_tokens)
        self.graphs: dict = {}
        self.pool = None
        # Updated in place, so describe() reads live counts.
        self.stats = {"buckets": list(self.lengths), "captured": [], "capture_seconds": 0.0,
                      "graph_calls": 0, "eager_calls": 0, "eager_reasons": {}}

    def _eager_reason(self, args: tuple, kwargs: dict) -> str | None:
        """Why a decoder call must run eagerly; None when it can replay a graph."""
        embeds = kwargs.get("inputs_embeds")
        if args or embeds is None or kwargs.get("input_ids") is not None:
            return "not_media"
        if set(kwargs) - _FORWARD_KEYS or kwargs.get("return_dict") is False:
            return "arguments"
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            return "cache"
        if torch.is_grad_enabled():
            return "gradient"
        if not embeds.is_cuda:
            return "device"
        if torch.cuda.is_current_stream_capturing():
            return "capturing"
        if embeds.ndim != 3 or embeds.shape[0] != 1:
            return "batch"
        positions, mask = kwargs.get("position_ids"), kwargs.get("attention_mask")
        length = embeds.shape[1]
        if (positions is None or positions.ndim != 3 or positions.shape[0] not in (3, 4)
                or positions.shape[1] != 1 or positions.shape[2] != length):
            return "positions"
        if length > self.lengths[-1]:
            return "too_long"
        if mask is not None and not bool(mask.all()):
            return "padding_mask"
        if positions.shape[0] == 4 and length > 1 and not bool((positions[0, 0].diff() == 1).all()):
            return "positions"   # row 0 drives the mask; eager inference would treat a jump as a packed boundary
        return None

    def __call__(self, *args, **kwargs):
        reason = self._eager_reason(args, kwargs)
        if reason is not None:
            if reason != "not_media":
                self.stats["eager_calls"] += 1
                self.stats["eager_reasons"][reason] = self.stats["eager_reasons"].get(reason, 0) + 1
            return self.eager(*args, **kwargs)
        embeds, positions = kwargs["inputs_embeds"], kwargs["position_ids"]
        length = embeds.shape[1]
        bucket = next(n for n in self.lengths if n >= length)
        key = (bucket, positions.shape[0], embeds.shape[2], embeds.dtype, positions.dtype)
        if key not in self.graphs:
            self._capture(key, embeds.device)
        graph, static_embeds, static_positions, out = self.graphs[key]
        static_embeds[:, :length].copy_(embeds)
        static_embeds[:, length:].zero_()
        static_positions[:, :, :length].copy_(positions)
        if bucket > length:
            tail = torch.arange(1, bucket - length + 1, device=positions.device, dtype=positions.dtype)
            static_positions[:, :, length:].copy_(positions[:, :, -1:] + tail.view(1, 1, -1))
        graph.replay()
        self.stats["graph_calls"] += 1
        from transformers.modeling_outputs import BaseModelOutputWithPast
        # A view of the static output: the caller reads it before the next replay overwrites it.
        return BaseModelOutputWithPast(last_hidden_state=out[:, :length])

    def _capture(self, key, device):
        from .cudagraphs import _eager_mask_decisions
        bucket, rows, width, dtype, position_dtype = key
        start = time.perf_counter()
        # Normal tensors even inside inference_mode: the static inputs are refilled in place on every replay.
        with torch.inference_mode(False), torch.no_grad():
            static_embeds = torch.zeros(1, bucket, width, dtype=dtype, device=device)
            static_positions = torch.arange(bucket, device=device, dtype=position_dtype).view(1, 1, -1).repeat(rows, 1, 1)
            kwargs = dict(input_ids=None, inputs_embeds=static_embeds, position_ids=static_positions,
                          attention_mask=None, past_key_values=None, use_cache=False)
            current = torch.cuda.current_stream(device)
            side = torch.cuda.Stream(device)
            side.wait_stream(current)
            with torch.cuda.stream(side):
                for _ in range(2):   # initialise lazily loaded kernels and allocator state outside the capture
                    self.eager(**kwargs)
            current.wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with _eager_mask_decisions(), torch.cuda.graph(graph, pool=self.pool):
                out = self.eager(**kwargs).last_hidden_state
        self.pool = graph.pool()
        self.graphs[key] = (graph, static_embeds, static_positions, out)
        self.stats["captured"].append(bucket)
        self.stats["capture_seconds"] = round(self.stats["capture_seconds"] + time.perf_counter() - start, 2)


def find_decoder(language_model: torch.nn.Module) -> torch.nn.Module:
    """The Transformers ``*TextModel`` decoder inside a (possibly PEFT-wrapped) language model."""
    for _, module in language_model.named_modules():
        if type(module).__name__.endswith("TextModel"):
            return module
    raise ValueError("media_cuda_graphs need a Transformers *TextModel decoder (Qwen vision backbones)")


def attach(language_model: torch.nn.Module, max_tokens: int = 2048) -> MediaGraphs:
    """Route the decoder's forward through MediaGraphs (idempotent) and return the wrapper."""
    decoder = find_decoder(language_model)
    if isinstance(decoder.forward, MediaGraphs):
        return decoder.forward
    graphs = MediaGraphs(decoder, max_tokens=max_tokens)
    decoder.forward = graphs
    return graphs
