# UAD + modality-conditioned ETGM/DLFM (V1.1)

For the full parameter, loss, learning-rate, file-change, and plan-alignment
audit in Chinese, see `CODE_CHANGE_EXPLANATION.md`.

This implementation follows `../idea.md` and retains the WHU-MARS data sampler,
ProCA/GPD implementation, retrieval evaluator, single-GPU training loop, and DDP
launcher. The original `configs/UAD.yml` remains the baseline.

## Runs

From `WHU/` with the original data and pretrained ViT path available, use the
same single-GPU recipe as `README.md` for WHU-MARS-1000:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/UAD_ViSA.yml \
  MODEL.DIST_TRAIN False DATASETS.DATASET_DIR WHU-MARS \
  SOLVER.IMS_PER_BATCH 64 SOLVER.BASE_LR 0.008 SOLVER.SEED 1234 \
  TEST.IMS_PER_BATCH 32 OUTPUT_DIR logs/UAD_ViSA_1000_single
```

For four-GPU DDP, follow the distinct `README.md` recipe. In particular,
`IMS_PER_BATCH` is the **global PKM sampler batch** and `BASE_LR` must be
overridden together with it:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py --config_file configs/UAD_ViSA.yml \
  MODEL.DIST_TRAIN True DATASETS.DATASET_DIR WHU-MARS \
  SOLVER.IMS_PER_BATCH 256 SOLVER.BASE_LR 0.032 SOLVER.SEED 1234 \
  TEST.IMS_PER_BATCH 32 OUTPUT_DIR logs/UAD_ViSA_1000_DDP
```

For the T0 control, replace `configs/UAD_ViSA.yml` by `configs/UAD_T0.yml`
and use a distinct output directory; keep the same batch/LR for its corresponding
single-GPU or DDP run. For WHU-MARS-2337 replace `DATASETS.DATASET_DIR` with
`WHU-MARS-2337` and use another output directory. The original `UAD.yml`
remains the B0 control, using the same per-run batch/LR as T0 and V1.1.
The original config specifies 64/0.008 and test batch 512; the README single
command overrides the test batch to 1024, while its DDP command overrides
training batch/LR to 256/0.032. For the full adapted model only, the validation
batch is 32 to limit graph memory (the eval features are batch independent).
At four GPUs, each rank receives 64 sampler
items; the existing loader expands the three modalities to 192 images per rank.
The PKM sampler batch remains global; each rank must receive at least two
identities and two samples per identity per modality for Specific WRT. `T0`
contains only the three input Specific Tokens and no additional loss, so it
isolates their effect on the UAD CLS. `full` adds the complete V1.1 path.

## Architecture

The pretrained UAD `pos_embed` stays `[1,129,768]`. A separate zero initialized
`specific_pos_embed` is inserted after its CLS position. Three distinct
`specific_tokens` start from the same random value. The selected token and CLS
traverse the original twelve ViT blocks once. Block 11 supplies 128 patches;
the final block and LN supply `f_s` and `f_p` (both 768D).

Shared ETGM reads `f_s.detach()`. RGB ETGM reads `f_p` for RGB, and IR ETGM
reads `f_p` for NIR/TIR. Each pool has eight experts and a separate router;
Top-2 weights combine the selected outputs. Every expert has 16 trainable
queries, its own linear condition projection, one 12-head self-attention, a
768->1536->768 GELU FFN, and three post-normalizations. Router balance uses
full-pool softmax importance, weighted by the active sample counts.

Each DLFM branch selects its own Top-40 block-11 patches by cosine score.
Each of 16 expert queries forms a 41-node graph with those patches. Patch node
values and adjacency are detached; query nodes retain gradients. Adjacency is
positive cosine similarity plus an identity self-loop, symmetrically degree
normalized in FP32. Independent shared and specific GCNs return only the query
node, followed by branch LN and query residual. A separate trainable `out_token`
and 32 updated queries enter one 12-head self-attention and a 768->3072->768
FFN. The first output is `f_l` (768D). Graph tensor dimensions are
`[B,16,41,768]` for each branch, with `B` the flattened per-rank image count.

The single retrieval vector is

```
z = cat(normalize(f_s), 0.5 * normalize(f_l)) / sqrt(1.25)  # 1536D
```

The model returns this vector for all evaluation modes, using each image's
recorded dataset modality ID (1 RGB, 2 NIR, 3 TIR), regardless of the evaluator's
group `mode`. `return_components=True` on an evaluation forward returns
`shared`, `local`, and `fusion` unit vectors for diagnostic retrieval using the
existing evaluator. The official result uses `fusion` only.

## Losses and gradients

The original UAD CE/WRT/ProCA/GPD code and coefficients remain in place,
receiving the original 768D pre-BN CLS. In `full` mode, only the following are
added, multiplied by `min(1, epoch/5)`:

```
0.05 * mean_m WRT(f_p[m], y[m])
+ 0.10 * CE(local_classifier(local_BN(f_l)), y.repeat(3))
+ 0.50 * WRT(cat(normalize(f_s).detach(), 0.5*normalize(f_l))/sqrt(1.25), y.repeat(3))
+ 0.0001 * router_importance_balance
```

Specific WRT consumes unnormalized pre-BN features, as does the original UAD
WRT. The Fusion loss sees exactly the evaluation vector's numerical values,
while its direct CLS segment has no gradient. Local loss can still train the
shared ViT through the Specific path. GPD's bank update remains after the
optimizer step. The existing checkpoint format saves only model weights, so a
finished adapted checkpoint is valid for evaluation; resuming GPD training
would also require recovering or rebuilding the prototype bank under a matched
baseline continuation protocol.

New parameters use 2x the corresponding original UAD LR rule, including bias
and optional classifier adjustments. Old parameters keep their prior rule.
Adapted runs keep every old parameter group's original UAD warmup and cosine
floor. Only new groups receive 2x the corresponding old-group warmup and floor;
the new/old ratio is therefore maintained without changing old bias or
classifier scheduling.

## Changed files

| File | Change |
| --- | --- |
| `model/backbones/vit_pytorch.py` | Optional Specific Tokens, separate position, block-11 patches, final CLS/Specific output. |
| `model/visa_adapt.py` | ETGM, sparse expert routing, DLFM, independent GCNs, shared fusion function. |
| `model/make_model.py` | `off`/`token_only`/`full` paths, Local head, modality-aware inference, strict adapted checkpoint loading. |
| `processor/processor.py` | Four added losses, loss ramp, evaluation modality forwarding; original ProCA/GPD loop retained. |
| `config/defaults.py` | Adaptation settings with default `off`. |
| `solver/make_optimizer.py` | 2x LR for new parameters. |
| `solver/scheduler_factory.py`, `solver/cosine_lr.py` | Adapted per-group warmup and LR floor. |
| `configs/UAD_T0.yml`, `configs/UAD_ViSA.yml` | Matched Token-only and complete experiment configurations; V1.1 validation batch 32. |
| `tests/test_visa_adapt.py` | CPU config parity, batch-independent inference, shape, gradient, fusion, and scheduler checks. |
| `CODE_CHANGE_EXPLANATION.md` | Detailed change record and plan-alignment audit. |

## Verification and limits

Run `python -m unittest discover -s tests -p 'test_visa_adapt.py' -v` from `WHU/`.
The tests exercise the graph and gradient semantics on smaller tensors; syntax
and YAML parsing can also be checked without data. The provided workspace has
no WHU-MARS images, pretrained weight file, or trained checkpoint. Full
single-GPU and NCCL-DDP training, throughput, and retrieval quality therefore
remain to be measured in the actual environment. Because the ViSA-style path
adds 24 experts and two 41-node graph branches, its compute and memory cost
should be measured before choosing a production batch size.
