"""Capture-only mask shortcuts must not escape into other model calls."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch

from jevany.cudagraphs import _eager_mask_decisions


@pytest.fixture
def masks(monkeypatch):
    from transformers import LlamaConfig, masking_utils

    config = LlamaConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, vocab_size=16)
    config._attn_implementation = "sdpa"
    embeddings = torch.zeros(1, 6, 8)
    originals = {name: getattr(masking_utils, name) for name in
                 ("find_packed_sequence_indices", "_ignore_causal_mask_sdpa", "is_tracing")
                 if hasattr(masking_utils, name)}
    # Restore upstream globals even if a regression leaves a wrapper installed.
    for name, original in originals.items():
        monkeypatch.setattr(masking_utils, name, original)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def mask(packed=False):
        positions = torch.tensor([[0, 1, 2, 4, 5, 6]]) if packed else torch.arange(6)[None]
        return masking_utils.create_causal_mask(
            config=config, inputs_embeds=embeddings, attention_mask=None,
            past_key_values=None, position_ids=positions,
        )

    return mask, masking_utils, originals


def test_overlapping_capture_contexts_restore_globals(masks):
    mask, masking_utils, originals = masks
    first, second = _eager_mask_decisions(), _eager_mask_decisions()
    first.__enter__()
    second.__enter__()
    try:
        assert mask() is None
    finally:
        first.__exit__(None, None, None)
        second.__exit__(None, None, None)
    for name, original in originals.items():
        assert getattr(masking_utils, name) is original
    assert mask(packed=True) is not None


def test_unrelated_capture_keeps_its_mask(masks):
    mask, _, _ = masks
    with ThreadPoolExecutor(max_workers=1) as pool, _eager_mask_decisions():
        assert mask() is None
        assert pool.submit(mask, True).result(timeout=5) is not None
        assert pool.submit(mask).result(timeout=5) is not None


def test_capture_failure_restores_nested_contexts(masks):
    mask, masking_utils, originals = masks
    with _eager_mask_decisions():
        with pytest.raises(RuntimeError, match="capture failed"):
            with _eager_mask_decisions():
                assert mask() is None
                raise RuntimeError("capture failed")
        assert mask() is None
    for name, original in originals.items():
        assert getattr(masking_utils, name) is original
    assert mask() is not None


def test_concurrent_capture_contexts_are_serialized(masks):
    mask, masking_utils, originals = masks
    attempting, entered = Event(), Event()

    def capture():
        attempting.set()
        with _eager_mask_decisions():
            entered.set()
            assert mask() is None

    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            with _eager_mask_decisions():
                future = pool.submit(capture)
                assert attempting.wait(5)
                assert not entered.wait(0.05)
                assert mask() is None
        finally:
            future.result(timeout=5)
    assert entered.is_set()
    for name, original in originals.items():
        assert getattr(masking_utils, name) is original


def test_causal_shortcut_does_not_change_other_threads_tracing(masks, monkeypatch):
    _, masking_utils, _ = masks
    if not hasattr(masking_utils, "_ignore_causal_mask_sdpa"):
        pytest.skip("this Transformers version does not use the tracing-dependent shortcut")
    original = masking_utils._ignore_causal_mask_sdpa
    entered, release = Event(), Event()

    def paused_shortcut(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(masking_utils, "_ignore_causal_mask_sdpa", paused_shortcut)

    def capture():
        with _eager_mask_decisions():
            return masking_utils._ignore_causal_mask_sdpa(None, 6, 6, 0, 0)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(capture)
        try:
            assert entered.wait(5)
            assert masking_utils.is_tracing()
        finally:
            release.set()
            assert future.result(timeout=5)
