# Concepts

This page explains the ideas and the vocabulary that the rest of the documentation assumes: what speculative decoding actually does, which number tells you whether a draft is any good, and what the tensors flowing through training mean. If you are looking for the commands instead, start with [Getting Started](getting_started.md); if you want to pick an algorithm, see the [Decision Guide](algorithms/decision_guide.md).

## Draft, verify, accept

Autoregressive decoding is slow for a reason that has nothing to do with arithmetic: generating one token requires one full forward pass of the target model, and that pass is bound by the memory bandwidth needed to stream the weights, not by the FLOPs it performs. Generating a token costs roughly the same as generating none. Verifying several tokens at once costs barely more than verifying one.

Speculative decoding exploits that asymmetry. A small **draft** model (also called the *drafter* or *speculator*) proposes the next `k` tokens cheaply. The target model — the **verifier** — then scores all `k` proposals in a single forward pass. The longest correct prefix is kept and the first wrong proposal is replaced with the verifier's own token, so decoding advances by somewhere between 1 and `k + 1` tokens per verifier pass.

The crucial property is that the output does not change. Verification compares against what the verifier would have produced anyway, and the first mismatch is resolved in the verifier's favour, so greedy speculative decoding returns exactly the greedy output of the target model alone. The draft can only affect *how fast* the answer arrives, never *what* the answer is. A badly trained draft makes generation slower, not wrong.

## Acceptance length is the metric that matters

Because a rejected proposal is simply discarded, the only thing a draft buys you is accepted tokens per verifier pass. That quantity is the **acceptance length**: the average number of tokens committed per verifier forward pass, counting the verifier's own token. An acceptance length of 1.0 means nothing is ever accepted and you are paying for the draft for no benefit. Published drafts in this repository typically report acceptance lengths between roughly 2 and 6 depending on model, algorithm, and domain.

Training reports two related per-step accuracies, logged as `full_acc_{k}` and `cond_acc_{k}` for each speculative step `k`:

- **`full_acc_k`** is the fraction of supervised positions where the draft's step-`k` prediction is correct *and* every earlier step was also correct. Correctness is chained, which is exactly how verification works: once a proposal is rejected the rest of the chain is thrown away no matter how good it was. This is the number that predicts acceptance.
- **`cond_acc_k`** is the fraction correct at step `k` among only those positions that survived step `k-1`. It isolates how well a single step does given a clean prefix, which is useful for diagnosing *where* a chain falls apart, but it overstates end-to-end behaviour.

Expected acceptance length follows from the chained accuracies — roughly `1 + Σ_k full_acc_k` — which is why `full_acc_0` dominates. A draft that is excellent at step 3 and mediocre at step 0 is a mediocre draft, because step 3 is only reached when step 0 was right.

Two further cautions when reading acceptance numbers. First, **acceptance length is not speedup**. Real speedup also depends on how much the draft itself costs per pass, on batch size, and on whether the server is latency-bound or throughput-bound; see [Throughput vs. Interactivity](tutorials/throughput_interactivity.md). Second, acceptance is strongly **domain-dependent**: predictable text such as code or structured output accepts far more readily than open-ended prose, so an acceptance number is only meaningful next to the dataset it was measured on.

## Why drafts read the verifier's hidden states

A draft that saw only the token ids would be a small language model guessing what a much larger one will say. The EAGLE family instead conditions the draft on the verifier's own internal activations, so the draft is extending a computation the verifier has already partly done rather than reproducing it from scratch.

Concretely, hidden states are captured from three of the verifier's decoder layers — low, middle, and late, selected by `eagle_aux_hidden_state_layer_ids` — and concatenated along the feature dimension. That is why the draft's first projection, `fc`, has shape `3 * target_hidden_size → hidden_size`: it fuses the three captured layers down to the draft's own width. Which three layers you pick measurably affects acceptance; see [Hidden State Layer ID Selection](hyperparameter_selection/hidden_state_layer_id_selection.md).

This conditioning is also why training needs a data-generation step at all. The hidden states have to come from somewhere, so training either reads them from a running vLLM instance or from pre-extracted files on disk. The `hs_connectors` package exists to move those tensors around, including across nodes.

## The draft vocabulary: `t2d` and `d2t`

A draft's output projection is over a vocabulary, and for a modern target model that vocabulary is large — often more than 128,000 tokens. For a small draft, that single `lm_head` matmul can dominate its cost. Speculators therefore lets a draft predict over a **reduced draft vocabulary** of the most frequent tokens (commonly 32,000), chosen from the token-frequency statistics that `prepare-data` collects.

Two tensors connect the two vocabularies, and the distinction matters because confusing them produces a draft that looks healthy and performs terribly:

- **`t2d`** ("target to draft") is a boolean mask over the *target* vocabulary, `True` at every target token id that made it into the draft vocabulary. It answers "is this target token predictable by the draft at all?" and is used to mask training targets.
- **`d2t`** ("draft to target") is an *offset* map over the *draft* vocabulary, not an index map. The target id for draft index `i` is `i + d2t[i]`. Storing offsets rather than absolute ids keeps the values small, and at inference the mapping back is a single add: `target_ids = draft_ids + d2t[draft_ids]`.

Because the selected token ids are sorted when the mapping is built, the two tensors encode the same selection and must agree:

```python
# The invariant every checkpoint's vocab mapping has to satisfy.
t2d.nonzero().squeeze(-1) == torch.arange(len(d2t)) + d2t
```

If they disagree, the draft's argmax is translated to the wrong target token, so nearly every proposal is rejected while every other signal — config, shapes, loading, loss curves from a previous run — still looks fine. This is not hypothetical: a published draft shipped with a `d2t` that had been round-tripped through bf16, which silently mapped 30,732 of its 32,000 draft ids to the wrong target id and dropped acceptance from 62% to 7%. `d2t` is an integer map and must never be stored in a floating-point dtype that cannot represent its values exactly.

Not every algorithm uses a reduced vocabulary. When a draft predicts over the full target vocabulary, neither tensor is present and no mapping is needed.

## Loss masks: what the draft is trained to predict

A draft should be trained on the tokens a model actually generates, not on the prompts it is given. Each training row therefore carries a **loss mask**: `0` over prompt tokens, `1` over the tokens the assistant produced. Only the masked-in positions contribute to the loss and to the accuracy metrics.

Rather than reimplementing every model's chat template, Speculators obtains these masks from the target model's own vLLM `/render` endpoint: each assistant turn is rendered with the serving template, and the tokens that rendering adds are the supervised ones. The mask is the definition of the boundary, which means the draft is trained on exactly the distribution it will be asked to predict at serving time, with no template configuration to get wrong.

## Sequence packing and `document_ids`

Training sequences vary wildly in length, and padding every batch to its longest member wastes a great deal of compute. Speculators packs several independent conversations end to end into one fixed-length sequence instead, using a bin-packing sampler to fill each batch efficiently.

Packing creates a hazard: positions that are adjacent in the packed tensor may belong to entirely unrelated conversations. A `document_ids` tensor records, for every position, which packed document it came from, with **`-1` marking padding**. Attention masks consult it so that a position can only attend within its own document. Any code that walks backwards or forwards from a position — anchor-based sampling, chained supervision — has to respect the same boundaries, because a chain that crosses one conditions the draft on a different conversation's tokens. Padding deserves the same care: `-1` is a sentinel, not a document, and indexing `document_ids` with a negative position silently wraps to the end of the sequence.

## Training data: offline and on-policy

Two shapes of training data appear throughout these docs.

**Off-policy** data uses the responses already present in a dataset. It is cheap, and it is what `prepare-data` consumes directly.

**On-policy** data replaces those responses with ones the *target model itself* generates, via `regenerate-responses`, before preparing them. It costs a generation pass over the dataset, but it trains the draft on the distribution it will actually face at serving time rather than on some other model's prose. Since acceptance depends on predicting the target's exact next token, that difference is usually worth the cost. Because regeneration over a large dataset is long-running, it supports `--resume`; see [Response Regeneration](../cli/response_regeneration.md).

## What a training run leaves behind

A training run writes a checkpoint directory containing one subdirectory per epoch, plus a `checkpoint_best` symlink pointing at the epoch with the lowest validation loss — that symlink, not the last epoch, is normally the model you want to serve. With `--save-best`, non-best epoch directories are pruned as training proceeds to save disk.

Alongside the epochs, the run records how to reproduce itself: `run.yaml` (the fully resolved config, re-runnable via `--config run.yaml`), `train_command.txt` (argv, git SHA, world size, package versions), and `speculators.patch` (any uncommitted diff). When publishing a model or a result, publish these too — an acceptance number without the command that produced it cannot be checked.

______________________________________________________________________

For the per-algorithm details behind these ideas, see [Algorithms](algorithms/index.md). For the end-to-end workflow, see the [Tutorials](tutorials/index.md).
