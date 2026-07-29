# Chapter 3 distillation-only experiments

This directory contains the pruning-free Chapter 3 student implementation.
The public experiment configs are in `avhubert/conf/distill_only/`; each
config composes only `base_continuous_75k.yaml` and materializes its own
architecture, initialization, noise, and schedule values.

The default main protocol is one continuous 75,000-update encoder run with
LR 0.002, 15,000 warm-up updates, Adam `(0.9, 0.999)`, zero weight decay,
polynomial power 1, and clip norm 10. S2 is the optional 50k+25k ordinary
distillation diagnostic. D/E/S selections are not made by the launcher:
update and commit the corresponding public YAML after the scientific
selection is frozen.

## Direct runner

Run commands from the repository root. `CH3_PROJECT_PATH` may override the
cluster checkout default.

```bash
sbatch scripts/run_distill_only.sh \
  --exp-name c3a_t12_historical_heads \
  --stage encoder

sbatch scripts/run_distill_only.sh \
  --exp-name c3a_t12_historical_heads \
  --stage export

sbatch scripts/run_distill_only.sh \
  --exp-name c3a_t12_historical_heads \
  --stage finetune \
  --run-name projection-fix-001

sbatch scripts/run_distill_only.sh \
  --exp-name c3a_t12_historical_heads \
  --stage evaluate \
  --run-name projection-fix-001 \
  --eval-name itut-screening-valid \
  --evaluation-phase screening \
  --eval-subsets valid
```

`--stage all` runs encoder, export, fine-tuning, and screening validation.
For S2 use `stage1` and then `stage2`; `all` performs both stages. Use
`--input-checkpoint` only with `finetune` or `evaluate` to select an explicit
existing checkpoint.

`--dry-run` prints exact commands without activating Conda or writing files.
Without `--resume`, a nonempty stage output is rejected. Fairseq training
resume requires that stage's `checkpoints/checkpoint_last.pt`. Export resume
strictly reloads the existing native student. Evaluation resume skips only a
condition whose exact result directory contains a `wer.*` file.

Outputs use:

```text
exp/chapter3_distill_only/<experiment>/seed_<seed>/
├── encoder/
├── export/student.pt
└── finetune_runs/<run-name>/
    ├── checkpoints/
    └── evaluations/<eval-name>/
```

Each executed stage writes an atomic `provenance.json` containing timestamps,
source branch/commit/tree and dirty state, the exact command, config and
checkpoint paths and hashes, resolved Hydra config when present, and exit
status. The simplified runner deliberately does not create source worktrees
or enforce the original commit when resuming; a changed or dirty checkout is
recorded and a dirty checkout emits a warning.

## Fine-tuning interface

The seq2seq wrapper infers the student encoder dimension and decoder embedding
dimension dynamically. If they differ, `encoder.proj` is a trainable linear
projection. During the first 48,000 fine-tuning updates the encoder backbone
is evaluated under `torch.no_grad()`, but this projection and the decoder
remain trainable. BatchNorm running statistics, dropout, and layerdrop retain
the historical training-mode behavior. Equal dimensions create no projection.

Although data, schedule, decoder, and augmentation arguments match the DP
fine-tuning workflow, fine-tuning and inference intentionally use:

```text
common.user_dir=<repository>/chapter3_distill_only
```

The native export has `_name: chapter3_av_hubert`; using only `avhubert/`
would not register that model and cannot reconstruct T12 or Conformer exports.

## Evaluation

`scripts/distill_only/evaluation_protocol_itut.yaml` is the only condition
definition. Screening validation is clean, babble 0 dB, and speech 0 dB.
Final evaluation is clean plus babble, music, and speech at -10, -5, 0, 5,
and 10 dB for the requested `valid`, `test`, or `valid,test` subsets.
Screening test evaluation is rejected. Clean decoding has no noise overrides;
noisy decoding uses probability 1 and the protocol's ITU mixing settings.

Evaluation labels are ordinary immutable-by-operator directories, not
digest-protected artifact identities. Test execution is operator-controlled;
this simplified runner has no automatic final-selection lock.

## Corrected fine-tuning reruns

The four projection-fix reruns should be launched separately:

```bash
for experiment in \
  c3b_t12_layer_to_layer \
  c3a_t12_historical_heads \
  c2_t6_historical_heads \
  b2_c1_t2_random_sequence
do
  sbatch scripts/run_distill_only.sh \
    --exp-name "$experiment" \
    --stage finetune \
    --run-name projection-fix-001
done
```

After each fine-tuning run completes:

```bash
sbatch scripts/run_distill_only.sh \
  --exp-name c3a_t12_historical_heads \
  --stage evaluate \
  --run-name projection-fix-001 \
  --eval-name itut-screening-valid \
  --evaluation-phase screening \
  --eval-subsets valid
```
