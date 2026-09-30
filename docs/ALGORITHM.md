# Algorithm

## Decision Model

JevAny receives a shared state and one or more typed questions. Each question defines its candidate answers explicitly. It does not decode an answer string.

The encoder reuses existing Qwen delimiters where available. Other tokenizers
receive five decision tokens with trainable embeddings. Both use the same layout:

```text
<state> state
<question> instruction <option> a </option> ... <decide>
```

For recurrent backbones such as Qwen3.8, sliding-window backbones, and architectures
without validated packed-mask support, each question is a separate causal row
that repeats the state. This prevents one question from changing another
question's representation. Supported full-attention backbones can use the
equivalent packed block-causal mask.

Let `h_d` be the hidden state at `<decide>` and `h_i` the hidden state at the end
of option `i`. The released pointer head computes

```text
q = Q(h_d), k_i = K(h_i)
z_i = k_i · q / sqrt(d_p) + R(k_i * q)
p_i = softmax(z)_i
```

Here `d_p=256`, `*` is elementwise multiplication, and `R` is a
256-wide Linear/GELU/Linear residual scorer with a zero-initialized output layer.
Temperature scaling divides `z` by a scalar fitted on a separate calibration
partition. It does not change the selected answer.

The direct-token readout assigns a fixed single-token label to each candidate.
At `<decide>`, it selects those labels' logits from the frozen LM output head
and normalizes them over the supplied candidates. It supports up to 255 choices.

## Supervised Fine-Tuning

For a hard target `y`, SFT minimizes `-log p_y`. A row may instead carry a
normalized soft target `t`, in which case the loss is `-Σ t_i log p_i`.
Pointer training normalizes probabilities over the candidate options;
direct-token training normalizes over the full vocabulary and uses the
candidates' label tokens as targets. Soft targets represent ambiguity or
missing information as a distribution over the options.

The released models use LoRA rank 8 over each backbone's supported
linear projections. Pointer checkpoints train the residual head with the adapter;
the direct-token checkpoint keeps the base LM head frozen. The base weights
remain frozen in both cases.

## RLCR

RLCR is an experimental continuation after SFT.

The paper *Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty* introduces Reinforcement Learning with Calibration Rewards. Its reward augments correctness with a proper scoring rule:

```text
r(c, q) = c - (q - c)²
```

Here `c` is answer correctness and `q` is confidence in the selected answer. A confidently correct decision approaches `1`; a confidently wrong decision approaches `-1`.

JevAny adapts that idea to a pointer model:

1. Compute the option logits `z`.
2. Draw a group of 32 zero-mean Gaussian perturbations around `z`.
3. For each proposal, take its highest-probability option and that option's probability as confidence.
4. Compute the RLCR reward and subtract the group mean to form advantages.
5. Apply a score-function gradient to the proposal distribution.
6. Weight the policy and supervised cross-entropy terms explicitly to control decision drift.

For proposal `z'`, the location score uses the isotropic Gaussian log density

```text
log q(z' | z) = constant - Σᵢ (z'ᵢ - zᵢ)² / (2σ²)
```

The sum is over option dimensions. Averaging over options would make the policy gradient weaker as the choice set grows, which is especially harmful for the many-choice tasks in the RL mixture. Gradients are clipped to norm 1 after distributed synchronization. Training logs the total objective, the policy term, cross-entropy, reward, Brier penalty, and the gradient norm before clipping.

The [starter RLCR recipe](../recipes/rlcr.toml) decays exploration standard
deviation from `0.4` to `0.1`, weights the policy term by `0.25`, and weights
the supervised anchor by `0.5`.

JevAny applies the paper's calibration reward and group-relative baseline to a policy over perturbed pointer logits. Optimization operates on these logits, without reasoning rollouts, confidence tokens, a critic, or a reference-model KL term.

## Training Data Strategy

We trained the released LoRA adapters with SFT on 1,772,725 text records containing
2,180,242 labelled decisions across preference, agent/tool decisions, reasoning,
classification and safety. See [DATA.md](DATA.md#current-release-scale) for the
data description and [the training guide](TRAINING.md) to build your own dataset.
