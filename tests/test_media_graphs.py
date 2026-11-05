"""Media CUDA graphs (jevany.media_graphs), the merged-load media dtype cast and their loading options."""
import argparse
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from jevany.checkpoint import Checkpoint, LoadOptions, cast_media_modules, load_options_from_args
from jevany.media_graphs import MediaGraphs, attach, find_decoder

WIDTH = 16


class ToyTextModel(nn.Module):
    """Causal one-head decoder with the Transformers text-model call signature; positions shift the embeddings."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.out = nn.Linear(WIDTH, WIDTH)

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None, attention_mask=None,
                past_key_values=None, use_cache=None, return_dict=None):
        x = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        if position_ids is not None:
            x = x + 0.01 * position_ids[-1].unsqueeze(-1).to(x.dtype)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        hidden = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return BaseModelOutputWithPast(last_hidden_state=self.out(hidden))


class ToyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyTextModel()


def media_call(length=5, rows=3, device="cpu", **overrides):
    kwargs = dict(input_ids=None, inputs_embeds=torch.randn(1, length, WIDTH, device=device),
                  position_ids=torch.arange(length, device=device).view(1, 1, -1).repeat(rows, 1, 1),
                  attention_mask=torch.ones(1, length, dtype=torch.long, device=device),
                  past_key_values=None, use_cache=False)
    kwargs.update(overrides)
    return kwargs


def test_media_graph_options_from_environment():
    assert not LoadOptions().media_cuda_graphs and LoadOptions().media_cuda_graph_max_tokens == 2048
    assert LoadOptions.from_env({}) == LoadOptions()
    options = LoadOptions.from_env({"JEVANY_MEDIA_CUDA_GRAPHS": "1", "JEVANY_MEDIA_CUDA_GRAPH_MAX_TOKENS": "1024"})
    assert options == LoadOptions(media_cuda_graphs=True, media_cuda_graph_max_tokens=1024)


@pytest.mark.parametrize("args,env,expected", [
    ({"media_cuda_graphs": True}, {"JEVANY_MEDIA_CUDA_GRAPHS": "0"}, LoadOptions(media_cuda_graphs=True)),
    ({"media_cuda_graph_max_tokens": 1024}, {"JEVANY_MEDIA_CUDA_GRAPH_MAX_TOKENS": "bad"},
     LoadOptions(media_cuda_graphs=True, media_cuda_graph_max_tokens=1024)),
    ({"media_cuda_graphs": True}, {"JEVANY_FUSED_KERNELS": "1", "JEVANY_MEDIA_CUDA_GRAPH_MAX_TOKENS": "512"},
     LoadOptions(media_cuda_graphs=True, media_cuda_graph_max_tokens=512, fused_kernels=True)),
])
def test_media_graph_flags_override_environment(args, env, expected):
    before = dict(env)
    assert load_options_from_args(argparse.Namespace(**args), env) == expected
    assert env == before


def test_media_graph_flags_keep_no_override_sentinel():
    args = argparse.Namespace(media_cuda_graphs=False, media_cuda_graph_max_tokens=None)
    assert load_options_from_args(args, {"JEVANY_MEDIA_CUDA_GRAPHS": "1"}) is None


@pytest.mark.parametrize("kwargs,message", [
    ({"media_cuda_graph_max_tokens": 127}, "at least 128"),
    ({"media_cuda_graph_max_tokens": 1024.0}, "at least 128"),
    ({"media_cuda_graphs": True, "compile_mode": "default"}, "enable one"),
    ({"media_cuda_graphs": True, "device_map": "auto"}, "one GPU"),
])
def test_media_graph_options_validate(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LoadOptions(**kwargs)


@pytest.mark.parametrize("device,multimodal,message", [("cpu", True, "CUDA only"), ("cuda", False, "multimodal")])
def test_media_graphs_fail_before_loading(device, multimodal, message):
    checkpoint = object.__new__(Checkpoint)
    checkpoint.meta = SimpleNamespace(multimodal=multimodal)
    with pytest.raises(ValueError, match=message):
        checkpoint.load(device, LoadOptions(media_cuda_graphs=True))


class ToyVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)


def test_cast_media_modules_casts_everything_outside_the_decoder():
    parent = nn.Module()
    parent.visual = ToyVisionModel()
    parent.language_model = nn.Linear(4, 4)
    parent.identity = nn.Identity()
    model = SimpleNamespace(mm=parent, lm=parent.language_model)
    assert cast_media_modules(model, torch.bfloat16) == ["visual"]
    assert parent.visual.proj.weight.dtype == torch.bfloat16
    assert parent.language_model.weight.dtype == torch.float32   # the decoder keeps the loader's own cast
    assert cast_media_modules(model, torch.bfloat16) == []
    assert cast_media_modules(SimpleNamespace(mm=None, lm=parent.language_model), torch.bfloat16) == []


def test_attach_wraps_the_text_model_once():
    language_model = ToyLanguageModel()
    graphs = attach(language_model, max_tokens=256)
    assert language_model.model.forward is graphs and attach(language_model) is graphs
    assert graphs.stats["buckets"] == [128, 192, 256]
    assert find_decoder(language_model) is language_model.model
    with pytest.raises(ValueError, match="TextModel"):
        find_decoder(nn.Linear(2, 2))
    with pytest.raises(ValueError, match="at least 128"):
        MediaGraphs(language_model.model, max_tokens=100)


def test_ineligible_calls_run_eagerly():
    torch.manual_seed(0)
    language_model = ToyLanguageModel()
    eager = language_model.model.forward
    graphs = attach(language_model)
    call = media_call()
    with torch.no_grad():
        expected = eager(**call).last_hidden_state
        torch.testing.assert_close(language_model.model(**call).last_hidden_state, expected)
        ids = torch.ones(1, 3, dtype=torch.long)
        torch.testing.assert_close(language_model.model(input_ids=ids).last_hidden_state,
                                   eager(input_ids=ids).last_hidden_state)
    language_model.model(**media_call())
    assert graphs.stats["graph_calls"] == 0 and graphs.stats["captured"] == []
    assert graphs.stats["eager_reasons"] == {"device": 1, "gradient": 1}   # text rows are not counted
    assert graphs.stats["eager_calls"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_media_graphs_match_eager_and_reuse_buckets():
    torch.manual_seed(0)
    language_model = ToyLanguageModel().cuda()
    eager = language_model.model.forward
    graphs = attach(language_model, max_tokens=256)
    with torch.no_grad():
        for length in (5, 130, 129):
            call = media_call(length, device="cuda")
            expected = eager(**call).last_hidden_state
            actual = language_model.model(**call).last_hidden_state
            assert actual.shape == expected.shape
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        language_model.model(**media_call(300, device="cuda"))
        padded = media_call(5, device="cuda")
        padded["attention_mask"][0, 0] = 0
        language_model.model(**padded)
        jump = media_call(5, rows=4, device="cuda")
        jump["position_ids"][0, 0, 3:] += 7
        language_model.model(**jump)
        language_model.model(**media_call(5, device="cuda", use_cache=True))
    with torch.inference_mode():
        call = media_call(7, device="cuda")
        torch.testing.assert_close(language_model.model(**call).last_hidden_state,
                                   eager(**call).last_hidden_state, rtol=1e-4, atol=1e-4)
        language_model.model(**media_call(200, device="cuda"))   # captured inside inference_mode
    with torch.no_grad():
        call = media_call(210, device="cuda")                     # refilled outside it
        torch.testing.assert_close(language_model.model(**call).last_hidden_state,
                                   eager(**call).last_hidden_state, rtol=1e-4, atol=1e-4)
    assert graphs.stats["captured"] == [128, 192, 256]
    assert graphs.stats["graph_calls"] == 6
    assert graphs.stats["eager_reasons"] == {"too_long": 1, "padding_mask": 1, "positions": 1, "cache": 1}
