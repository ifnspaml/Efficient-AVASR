# Chapter 3 v3 experiments

Scientific design: [PhD thesis issue 12, comment 4905908061](https://github.com/joyolee/PhD-Thesis-Zhengyang-Li-2026/issues/12#issue-4905908061).

This document is the operational guide for branch `ch3-distill-only-v3`. New
configs live only in `avhubert/conf/distill_v3`, new launchers live only in
`scripts_v3`, and results live under `exp/chapter3_distill_only`. The historical
`exp/chapter3_distill_only_preflight` directory is read-only and is never
searched automatically.

## Audit

The retained distillation-only implementation supports historical final-output
prediction heads, layer-to-layer adapters, configurable teacher targets and
student mappings, train-only encoder noise, continuous 75k training, and both
teacher-sequence and random-sequence initialization.

The joint-DP criterion supports `raw` and `log_sig` cosine losses and arbitrary
nonnegative L1 weights. The v3 launcher rejects negative or non-finite weights.
The legacy 75k joint-DP recipe is not loss-homogeneous: `resnet.yaml` uses raw
cosine with L1 0.1 for the first 50k updates, while `distill.yaml` uses raw
cosine with L1 1.0 for the final 25k. K-series pairs correct this by keeping one
loss across both stages.

All v3 downstream runs call `scripts_v3/distill_only/run_finetune.sh`, which
explicitly fixes 60k updates, a 48k encoder freeze, LR 0.0005, clip norm 0,
update frequency 8, MUSAN noise probability 0.25, and SNR 0 dB. A4 is not run
or claimed comparable until its protocol is decided by the user.

The v3 launchers deliberately contain no Git state, file digest, source
snapshot, or immutable-manifest enforcement. Resume is explicit and depends on
the expected checkpoint being present and loadable. Evaluation resume skips a
condition only when its `wer.*` output exists.

## Experiment order and selection gates

1. A0--A3 compare initialization and capacity allocation. A4 is deferred.
2. L0--L6 study supervision using the provisional T12 winner. Do not submit
   this T12 series as final if Stage A selects another architecture.
3. AL is created only after A and L results, using names
   `al0_<architecture>_<selected_loss_tag>` and
   `al1_<architecture>_<selected_loss_tag>`.
4. K0 versus K1 selects cosine type; submit only the matching K2 and K3
   candidates to select the L1 weight.
5. J1--J3 reuse the selected K loss via launcher arguments.
6. Stage F evaluates manually frozen checkpoint paths from a populated copy of
   `avhubert/conf/distill_v3/final_models.example.yaml`.
7. Stage X remains phase-gated; only target/reset/budget utilities are present.

## A-series

Run from the repository root:

```bash
for experiment in \
  a0_t2_teacher_init_noisy \
  a1_t2_random_sequence_noisy \
  a2_t6_historical_heads_noisy \
  a3_t12_historical_heads_noisy
do
  sbatch --export=ALL,CH3_PROJECT_PATH="$PWD" \
    scripts_v3/run_ch3_distill_only.sh \
    --exp-name "$experiment" \
    --stage all \
    --seed 1337 \
    --run-name lr5e-4 \
    --finetune-lr 0.0005
done
```

Replace `sbatch` with `bash` and add `--dry-run` to inspect commands without
creating outputs. Resume an interrupted pipeline by adding `--resume`, or
resume one explicit stage with `--stage <stage> --resume`.

## L-series

```bash
for experiment in \
  l0_t12_historical_heads_noisy \
  l1_t12_historical_heads_noisy_targets4812 \
  l2_t12_historical_heads_noisy_targets812 \
  l3_t12_historical_heads_noisy_target12 \
  l4_t12_layer_to_layer_noisy \
  l5_t12_layer_to_layer_noisy_target12 \
  l6_t12_historical_heads_clean
do
  sbatch --export=ALL,CH3_PROJECT_PATH="$PWD" \
    scripts_v3/run_ch3_distill_only.sh \
    --exp-name "$experiment" \
    --stage all \
    --seed 1337 \
    --run-name lr5e-4 \
    --finetune-lr 0.0005
done
```

## K-series

```bash
sbatch --export=ALL,CH3_PROJECT_PATH="$PWD" \
  scripts_v3/run_ch3_joint_dp.sh \
  --exp-name k0_hybrid_tau70_raw_cos_l1_0p1 \
  --stage all --seed 1337 --run-name lr5e-4 --finetune-lr 0.0005

sbatch --export=ALL,CH3_PROJECT_PATH="$PWD" \
  scripts_v3/run_ch3_joint_dp.sh \
  --exp-name k1_hybrid_tau70_log_sig_cos_l1_0p1 \
  --stage all --seed 1337 --run-name lr5e-4 --finetune-lr 0.0005
```

After K0/K1 validation, choose `raw` or `log_sig` and submit exactly the K2 and
K3 filenames containing that cosine tag. The loss is encoded in both stage
configs; optional launcher loss arguments must agree with it.

## J-series

After selecting the K loss, replace the example values below consistently:

```bash
for experiment in j1_transformer_tau65 j2_hybrid_tau70 j3_hybrid_tau80
do
  sbatch --export=ALL,CH3_PROJECT_PATH="$PWD" \
    scripts_v3/run_ch3_joint_dp.sh \
    --exp-name "$experiment" \
    --stage all \
    --seed 1337 \
    --run-name lr5e-4 \
    --finetune-lr 0.0005 \
    --cosine-type log_sig \
    --l1-weight 0.1
done
```

J commands fail before creating outputs if either loss argument is missing.

## Individual stages and explicit checkpoints

```bash
scripts_v3/run_ch3_distill_only.sh \
  --exp-name a3_t12_historical_heads_noisy \
  --stage export --input-checkpoint /path/to/checkpoint_last.pt --dry-run

scripts_v3/run_ch3_distill_only.sh \
  --exp-name a3_t12_historical_heads_noisy \
  --stage finetune --input-checkpoint /path/to/student.pt \
  --run-name lr5e-4 --finetune-lr 0.0005 --dry-run

scripts_v3/run_ch3_joint_dp.sh \
  --exp-name k0_hybrid_tau70_raw_cos_l1_0p1 \
  --stage post_distill --input-checkpoint /path/to/pruned-student.pt --dry-run
```

An explicit path may point into preflight storage; no path there is inferred.
`--input-checkpoint` is rejected with `--stage all` because its parent stage
would be ambiguous.

## Evaluation

Screening validation:

```bash
sbatch --export=ALL,CH3_PROJECT_PATH="$PWD" \
  scripts_v3/run_ch3_distill_only.sh \
  --exp-name a3_t12_historical_heads_noisy \
  --stage evaluate --run-name lr5e-4 \
  --evaluation-phase screening --eval-subsets valid \
  --eval-name screening
```

Final valid/test evaluation of frozen selections:

```bash
cp avhubert/conf/distill_v3/final_models.example.yaml /path/to/final_models.yaml
# Populate only the selected checkpoint paths, then:
sbatch --export=ALL,CH3_PROJECT_PATH="$PWD" \
  scripts_v3/run_ch3_final_evaluation.sh \
  --registry /path/to/final_models.yaml \
  --models f1_best_manual,f2_best_wer_joint_dp \
  --eval-name itut-final
```

The final matrix contains clean plus babble, speech, and music at -10, -5, 0,
5, and 10 dB on both validation and test.

## Stage-X utilities

```bash
python scripts_v3/distill_only/stage_x_tools.py \
  --project-path "$PWD" resolve-targets --checkpoint /path/to/large.pt

python scripts_v3/distill_only/stage_x_tools.py \
  --project-path "$PWD" reset-encoder \
  --checkpoint /path/to/materialized.pt --output /path/to/reset.pt --seed 1337

python scripts_v3/distill_only/stage_x_tools.py \
  --project-path "$PWD" budget-report \
  --candidate /path/to/candidate.pt --reference /path/to/x0.pt \
  --parameter-tolerance 0.05 --flop-tolerance 0.05
```

No runnable X config is created until base winners, the large teacher, and the
deployment budget are frozen.
