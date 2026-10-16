"""Checkpoint cadence, optimizer-step recovery and incomplete random state."""
import json

import pytest
import torch
from safetensors.torch import load_file

from jevany.checkpoint import Checkpoint
from jevany.train import main
from test_backbones import make_base


@pytest.mark.parametrize("objective", ["sft", "rlcr"])
def test_checkpoint_cadence_and_each_unfinished_step_preserve_training(tmp_path, objective):
    base = tmp_path / "base"
    make_base(base, "llama")
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps({"state": "state " * (2 + index), "questions": {
        "q": {"type": "choice", "instructions": "choose",
              "criteria": {"yes": None, "no": None, "other": None},
              "label": ["yes", "no", "other"][index % 3]},
    }}) + "\n" for index in range(5)))
    args = ["--base", str(base), "--data", str(data), "--device", "cpu", "--lora", "2",
            "--head-dim", "8", "--epochs", "2", "--batch", "1", "--accum", "2",
            "--seed", "3", "--lr", "0.01", "--lora-dropout", "0.2"]
    if objective == "rlcr":
        args += ["--rlcr", "--rlcr-group-size", "4"]
    plain = main(args + ["--out", str(tmp_path / "plain")])
    assert not (plain / "checkpoints").exists()
    full = main(args + ["--out", str(tmp_path / "full"), "--checkpoint-every-steps", "1"])

    def weights(directory):
        return load_file(directory / "adapter_model.safetensors"), Checkpoint(directory).meta.head

    expected = weights(full)
    for actual, reference in zip(weights(plain), expected):
        assert actual.keys() == reference.keys()
        assert all(torch.equal(actual[key], reference[key]) for key in reference)
    states = sorted((full / "checkpoints").glob("step-*"))
    assert [path.name for path in states] == [f"step-{index:06d}" for index in range(1, 7)]
    metrics = json.loads((full / "training_metrics.json").read_text())
    for state in states[:-1]:
        resumed = main(["--resume", str(state), "--out", str(tmp_path / f"resumed-{state.name}")])
        for actual, reference in zip(weights(resumed), expected):
            assert actual.keys() == reference.keys()
            for key in reference:
                assert torch.equal(actual[key], reference[key]), (state.name, key)
        resumed_metrics = json.loads((resumed / "training_metrics.json").read_text())
        for key in ("optimizer_steps", "records_seen", "forward_tokens"):
            assert resumed_metrics[key] == metrics[key], (state.name, key)
        config = json.loads((resumed / "training_config.json").read_text())
        assert config["resume_from"] == str(state.resolve())
    with pytest.raises(ValueError, match="already finished"):
        main(["--resume", str(states[-1]), "--out", str(tmp_path / "finished")])
    assert not (tmp_path / "finished").exists()


@pytest.mark.parametrize("missing", ["rank_rng", "rank", "rank_state", "torch"])
def test_incomplete_rng_state_is_rejected_before_creating_output(tiny_run, tmp_path, missing):
    root, _, _ = tiny_run
    original = main([
        "--base", str(root / "base"), "--data", str(root / "data" / "train.jsonl"),
        "--out", str(tmp_path / "original"), "--device", "cpu", "--lora", "2",
        "--head-dim", "8", "--lora-targets", "qv", "--max-steps", "2", "--accum", "1",
        "--checkpoint-every-steps", "1",
    ])
    checkpoint = original / "checkpoints" / "step-000001"
    state = torch.load(checkpoint / "trainer_state.pt", weights_only=True)
    if missing == "rank_rng":
        state.pop("rank_rng")
    elif missing == "rank":
        state["rank_rng"] = []
    elif missing == "rank_state":
        state["rank_rng"] = [None]
    else:
        state["rank_rng"][0].pop(missing)
    torch.save(state, checkpoint / "trainer_state.pt")
    output = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="incomplete.*RNG state"):
        main(["--resume", str(checkpoint), "--out", str(output)])
    assert not output.exists()
