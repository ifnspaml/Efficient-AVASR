# Chapter 3 Stage-K joint distillation and pruning

This document defines only the unrun Stage-K matrix. Completed A- and L-series
configs, launchers, checkpoints, result directories, and numerical paths are
not inputs to this launcher and are not modified by this design.

Scientific design: [PhD thesis issue 12](https://github.com/joyolee/PhD-Thesis-Zhengyang-Li-2026/issues/12).

## Prediction-head equivalence audit

The completed Stage-L `historical_pred_heads` and legacy joint-DP `predlayer`
paths are not equivalent.

| Property | Stage-L `historical_pred_heads` | Legacy joint-DP `predlayer` | New joint-DP `historical_pred_heads` |
|---|---|---|---|
| Student input | True final encoder output | `extract_intermediate_features()[-1]` | True final encoder output from `extract_features(mask=False)` |
| Target source | Selected entries from teacher `extract_intermediate_features()` | Same | Same |
| All targets predicted from final student state | Yes | Yes | Yes |
| Target independence | Independent expansion chunks and independent split projections | Independent per-target modules | Same as Stage L |
| Projection | `Linear(Ds, N*H)` → GELU → per-target `Linear(H, Dt)` | Per-target `Linear(Ds, Dt)` → GELU | Same as Stage L |
| Hidden width | `H=Ds` when unspecified | No hidden expansion | `H=Ds` |
| Initialization | PyTorch `nn.Linear` defaults for expansion; uniform ±`H^-0.5` for split weights and biases | PyTorch `nn.Linear` defaults | Same as Stage L |
| Normalization in head | None | None | None |
| Output shape | `B × N × T × Dt`, contiguous | Stack to `B × N × T × Dt` | Same as Stage L |
| Layer 0 meaning | Fused frontend representation before positional convolution/Transformer layer 1 | Same teacher indexing | Same teacher indexing |
| Layer `i>0` meaning | Output of Transformer layer `i` | Same | Same |

The new option is additive. Legacy `predlayer` and `layer2layer` branches are
unchanged. The Stage-L module is also unchanged. A separate implementation in
`avhubert/historical_prediction_heads.py` avoids importing or changing the
completed distillation-only runtime; an equivalence test fixes the same random
seed and verifies identical parameters and outputs against the Stage-L class.

The head performs no extra normalization. The unchanged criterion compares the
`B × N × T × D` tensors directly. Its formulas remain:

- raw cosine: `-mean(cosine_similarity(student, teacher))`;
- log-sigmoid cosine: `-mean(log(sigmoid(cosine_similarity(student, teacher))))`.

The existing joint-DP criterion already logs total loss, raw L1, raw cosine,
regularization, expected/target sparsity, and Lagrange multipliers. Hydra stores
the resolved configuration. Physical pruning additionally writes total and
component realized sparsity into `run_metadata.json`.

## Corrected matrix

| ID | Stage 1 config | Stage 2 config | Matching | Teacher targets | Cosine | L1 Stage 1 | L1 Stage 2 |
|---|---|---|---|---|---:|---:|---:|
| K-ref | `kref_l2l_t04812_raw_l1_0p1.yaml` | `kref_l2l_t04812_raw_l1_1p0.yaml` | layer-to-layer | `{0,4,8,12}` | raw | 0.1 | 1.0 |
| K-T | `kt_l2l_t812_raw_l1_0p1.yaml` | `kt_l2l_t812_raw_l1_1p0.yaml` | layer-to-layer | `{8,12}` | raw | 0.1 | 1.0 |
| K0 | `k0_pred_t812_raw_l1_0p1.yaml` | `k0_pred_t812_raw_l1_1p0.yaml` | historical prediction heads | `{8,12}` | raw | 0.1 | 1.0 |
| K1 | `k1_pred_t812_log_sig_l1_0p1.yaml` | `k1_pred_t812_log_sig_l1_1p0.yaml` | historical prediction heads | `{8,12}` | log_sig | 0.1 | 1.0 |
| K2 | `k2_pred_t812_raw_l1_0p1.yaml` | `k2_pred_t812_raw_l1_0p1.yaml` | historical prediction heads | `{8,12}` | raw | 0.1 | 0.1 |
| K3 | `k3_pred_t812_log_sig_l1_0p1.yaml` | `k3_pred_t812_log_sig_l1_0p1.yaml` | historical prediction heads | `{8,12}` | log_sig | 0.1 | 0.1 |

All Stage-1 configs inherit the unchanged v3 50k base and set hybrid
`conv,head,interm` pruning at target sparsity 0.70. All Stage-2 configs inherit
the unchanged v3 25k post-distillation base. Both stages use seed 1337,
`clip_norm=10`, `update_freq=4`, and training noise probability 0.25 at 0 dB.

The controlled comparisons are:

1. K-ref vs K-T: target set only, `{0,4,8,12}` vs `{8,12}` under layer-to-layer matching.
2. K-T vs K0: matching only, layer-to-layer vs historical prediction heads at `{8,12}`.
3. K-ref vs K0: full transfer of the Stage-L target and prediction-head principle.
4. K0 vs K1: raw vs log_sig cosine with L1 `0.1 → 1.0`.
5. K0 vs K2: Stage-2 L1 1.0 vs 0.1 with raw cosine.
6. K1 vs K3: Stage-2 L1 1.0 vs 0.1 with log_sig cosine.
7. K2 vs K3: raw vs log_sig cosine with L1 `0.1 → 0.1`.

Automated config tests compare fully composed dictionaries and enforce exactly
those field differences.

## Historical K-ref encoder audit and reuse

K-ref is the historical `exp/distill/resnet` encoder recipe, not a new 75k
encoder-learning run. The audit used the launcher, the stored resolved Hydra
snapshots, the completion logs, and the checkpoint chain:

| Property | Historical resolved value |
|---|---|
| Matching / targets | `layer2layer`, `0.4,8,12` (frontend 0 plus Transformer layers 4, 8, and 12) |
| Stage-1 loss | raw cosine, L1 0.1 |
| Stage-2 loss | raw cosine, L1 1.0 |
| Pruning | `conv,head,interm`, target sparsity 0.70, 10k sparsity warm-up |
| Budget | 50k joint DP + materialization + 25k post-distillation |
| Optimization | the same composite Stage-1 and Adam/polynomial Stage-2 settings; launcher-resolved `update_freq=[4]`, `clip_norm=10.0` |
| Teacher / initialization | `base_vox_iter5.pt` for both teacher and initial student |
| Data / noise / seed | LRS3 433h, MUSAN training noise probability 0.25 at 0 dB, seed 1337 |

The logs record termination exactly at 50,000 and 25,000 updates. The final
materialized encoder is loadable by the current Stage-K runtime:

```text
exp/distill/resnet/final/checkpoints/pruned_checkpoint_last_final.pt
```

The Stage-K launcher treats this historical chain as read-only. For `k-ref`,
`--stage all` skips joint DP, pruning, post-distillation, and export, then runs
the common Chapter-3 fine-tuning and evaluation stages. Direct K-ref encoder
stage requests are rejected to prevent accidental retraining. An alternative
historical root can be audited/debugged with `--historical-kref-root PATH` or
`CH3_HISTORICAL_KREF_ROOT`; it does not change the K-ref scientific definition.

The historical run did not record its Git SHA. Its stored resolved configs,
logs, checkpoint timestamps, and loadability establish protocol/checkpoint
provenance, but the missing original SHA remains a reproducibility caveat.

The existing `exp/finetune/asr/resnet_lr5e-4` downstream run also resolves to
60k updates, a 48k encoder freeze, LR 0.0005, `update_freq=[8]`, clip norm 0.0,
noise probability 0.25 at 0 dB, and seed 1337. Its stored ITU-T outputs cover
validation and test for babble, speech, and music at -10, -5, 0, +5, and
+10 dB. It is numerically protocol-compatible, but it predates Stage-K's
isolated metadata and validation-screening workflow. The command below reruns
the matched downstream path inside `exp/chapter3_joint_dp/k_ref/` so selection
can be recorded from validation screening before test reporting.

## Launcher and isolated outputs

The dedicated interface is `scripts_v3/run_ch3_joint_dp_stage_k.sh`. It never
calls `scripts/run_pruning.sh` and never writes to an A/L or historical joint-DP
result directory. The historical K-ref directory is a read-only encoder input.
New outputs are isolated as:

```text
exp/chapter3_joint_dp/
  k_ref|k_t|k0|k1|k2|k3/
    seed_1337/
      joint_dp/
      pruned/
      post_distill/
      export/
      finetune_runs/<run-name>/
      run_metadata.json
```

Nonempty stage destinations are rejected unless `--resume` is explicit. A
nonempty experiment root without compatible Stage-K metadata is always
rejected. The metadata records the Git branch/SHA/dirty flag, experiment and
seed, both config paths, resolved Hydra output paths, every checkpoint, fixed
protocol, fine-tuning config, and evaluation selection policy.

Dry-run all six without submitting GPU jobs. K-ref prints the historical
checkpoint chain and skips encoder-learning commands:

```bash
for experiment in k-ref k-t k0 k1 k2 k3; do
  bash scripts_v3/run_ch3_joint_dp_stage_k.sh \
    --experiment "$experiment" --stage all --seed 1337 \
    --finetune-run-name lr5e-4 --finetune-lr 0.0005 --dry-run
done
```

Submit one of the five new encoder-learning experiments only after reviewing
its dry-run:

```bash
sbatch --export=ALL,CH3_PROJECT_PATH="$PWD" \
  scripts_v3/run_ch3_joint_dp_stage_k.sh \
  --experiment k-t --stage all --seed 1337 \
  --finetune-run-name lr5e-4 --finetune-lr 0.0005
```

Individual stages are `joint_dp`, `prune`, `post_distill`, `export` (alias
`save`), `finetune`, and `evaluate`. Stage-specific checkpoint override flags
are listed by `--help`; overrides and any non-default fine-tuning LR are
printed and recorded.

Run matched K-ref downstream fine-tuning and validation screening from the
reused historical encoder:

```bash
sbatch --export=ALL,CH3_PROJECT_PATH="$PWD" \
  scripts_v3/run_ch3_joint_dp_stage_k.sh \
  --experiment k-ref --stage all --seed 1337 \
  --finetune-run-name lr5e-4 --finetune-lr 0.0005 \
  --evaluation-phase screening --evaluation-subsets valid \
  --evaluation-name screening-valid
```

## Fine-tuning and evaluation

All K runs delegate to the common Chapter-3 fine-tuning runner with 60k
updates, a 48k encoder freeze, peak LR 0.0005, update frequency 8, clip norm
0.0, and training noise probability 0.25 at 0 dB. The launcher does not inherit
the historical 0.001 joint-DP fine-tuning LR.

Validation-only screening (the only model-selection evaluation) is:

```bash
sbatch --export=ALL,CH3_PROJECT_PATH="$PWD" \
  scripts_v3/run_ch3_joint_dp_stage_k.sh \
  --experiment k0 --stage evaluate --seed 1337 \
  --finetune-run-name lr5e-4 \
  --evaluation-phase screening --evaluation-subsets valid \
  --evaluation-name screening-valid
```

It evaluates clean, babble 0 dB, and speech/second-talker 0 dB. Test WER is
never a selection input.

After selecting a model from validation results, final reporting is:

```bash
sbatch --export=ALL,CH3_PROJECT_PATH="$PWD" \
  scripts_v3/run_ch3_joint_dp_stage_k.sh \
  --experiment k0 --stage evaluate --seed 1337 \
  --finetune-run-name lr5e-4 \
  --evaluation-phase final --evaluation-subsets valid,test \
  --evaluation-name itut-final
```

This reuses the Chapter-3 ITU-T implementation for clean plus babble, speech,
and music at -10, -5, 0, +5, and +10 dB on validation and test. Test results
are reporting-only.
