# Chapter 3 distillation-only execution

All commands below run from the repository root on
`feat/ch3-distill-only`. For a new run, the launcher requires a fully clean
development checkout and verifies the fixed branch ancestor, teacher, LRS3
manifests, tokenizer, and MUSAN training manifest by SHA256.

## Fine-tuning encoder/decoder interface

The seq2seq encoder backbone remains frozen for updates 0–47,999 and unfreezes
at update 48,000. The encoder/decoder interface is handled separately:

- the encoder output dimension is read from
  `w2v_model.encoder.embedding_dim`;
- the decoder input dimension is read from `cfg.decoder_embed_dim`;
- equal dimensions add no parameters;
- unequal dimensions create
  `Linear(encoder_output_dim, decoder_input_dim)`;
- this projection is outside the frozen-backbone `torch.no_grad()` region, so
  it and the decoder train from update zero;
- its checkpoint keys remain `encoder.proj.weight` and
  `encoder.proj.bias`.

The refactor does not put the encoder into evaluation mode. ResNet BatchNorm
running statistics, dropout, and layerdrop therefore retain the historical
joint-DP fine-tuning behavior while backbone parameter gradients remain
disabled during the freeze.

Inspect local exported C2/C3 interfaces without modifying their artifacts:

```bash
python scripts/distill_only/inspect_seq2seq_interface.py \
  exp/chapter3_distill_only/c2_t6_historical_heads \
  exp/chapter3_distill_only/c3a_t12_historical_heads
```

## Commit-pinned source worktrees

Preparing a new run creates a detached worktree at the clean development
checkout's exact commit. The default root is derived from the repository
parent:

```text
<repo-parent>/<repo-name>_worktrees/chapter3_distill_only/
```

Override it when necessary:

```bash
scripts/distill_only/launch.sh \
  --experiment c2_t6_historical_heads \
  --stage prepare \
  --source-worktree-root ../chapter3-source-worktrees
```

The manifest's immutable `source_snapshot` records the repository root,
detached-worktree path, commit SHA, tree SHA, integrity digest, and creation
timestamp. Encoder training, export, profiling, fine-tuning, and decoding use
that worktree as `cwd`; its repository and vendored Fairseq paths are
prepended to `PYTHONPATH`. The launcher verifies commit, tree, digest, tracked
state, staged state, and unexpected untracked files before and after every
stage.

Inspect the source bound to a run:

```bash
manifest=exp/chapter3_distill_only/c2_t6_historical_heads/seed_1337/manifest.v1.json
jq '.immutable.source_snapshot' "$manifest"
jq '.runtime.stage_provenance' "$manifest"
```

Development may continue and be committed in the main checkout after
preparation. Resume from either the main checkout or a generated pinned SLURM
script; the run continues at the manifest commit:

```bash
scripts/distill_only/launch.sh \
  --experiment c2_t6_historical_heads \
  --stage export

sbatch \
  exp/chapter3_distill_only/c2_t6_historical_heads/seed_1337/source/slurm/export.slurm
```

Legacy manifests without `source_snapshot` are readable, but execution stops
with a compatibility error rather than assigning them to the current HEAD.
Start a new run unless an explicit, reviewed migration from the recorded
commit is performed.

Pinned worktrees are intentionally retained for later stages and resumption.
After a run is complete and no longer needs to resume, inspect it, then remove
it explicitly from the repository that created it:

```bash
worktree=$(jq -r '.immutable.source_snapshot.worktree_path' "$manifest")
git worktree remove "$worktree"
git worktree prune
```

Never remove a worktree for an active or resumable experiment.

## Verification gates

```bash
env PYTHONPATH=fairseq:. PYTHONPYCACHEPREFIX=/tmp/ch3-pyc \
  /home/zhengyangli/work_fast/envs/dpavhubert_pro6000/bin/python \
  -m unittest discover -s tests/distill_only -p 'test_*.py' -v

env PYTHONPATH=fairseq:. PYTHONPYCACHEPREFIX=/tmp/ch3-pyc \
  /home/zhengyangli/work_fast/envs/dpavhubert_pro6000/bin/python \
  tests/distill_only/smoke_forward_backward.py \
  --config-name c3b_t12_layer_to_layer --device cuda
```

Use the same smoke command for B2/C1, C2, C3a, and D2. It performs an
optimizer update, strict model/optimizer checkpoint round-trip, and a resumed
update while asserting that the teacher receives no gradients. The
construction smoke accepts `--verify-export` and reloads the teacher/head-free
student both directly through Fairseq and through the unchanged
`HubertEncoder` path.

Run the heavier trainer-level checkpoint gate explicitly on CPU or GPU:

```bash
env PYTHONPATH=fairseq:. PYTHONPYCACHEPREFIX=/tmp/ch3-pyc \
  /home/zhengyangli/work_fast/envs/dpavhubert_pro6000/bin/python \
  tests/distill_only/smoke_trainer_checkpoint_resume.py \
  --config-name b2_c1_t2_random_sequence --device cuda
```

This uses Fairseq's real `Trainer`, `checkpoint_utils`, LRS3 train iterator,
and frozen teacher checkpoint. It checks exact in-stage restoration of model,
Adam, scheduler/LR/update, iterator offset, and meters, then checks that S2's
`finetune_from_model` retains student/head weights while resetting optimizer,
scheduler, update count, iterator, and meters. It is opt-in because it builds
the teacher/student three times and writes a complete temporary checkpoint.
Pass an empty `--work-dir` to retain the diagnostic checkpoint and output
directory.

## B/C runs

Inspect a complete command without creating output:

```bash
scripts/distill_only/launch.sh \
  --experiment c2_t6_historical_heads --stage all --dry-run
```

Launch B1, B2/C1, C2, C3a, and C3b by changing `--experiment` and removing
`--dry-run`. The default is seed 1337, one GPU, `max_tokens=4000`, and
`update_freq=4`. If memory requires `max_tokens=2000`, pass
`--max-tokens 2000 --update-freq 8`; preflight rejects any pair whose product
is not 16,000.

Generate the validation-only C report:

```bash
scripts/distill_only/report.py \
  exp/chapter3_distill_only/b2_c1_t2_random_sequence/seed_1337/manifest.v1.json \
  exp/chapter3_distill_only/c2_t6_historical_heads/seed_1337/manifest.v1.json \
  exp/chapter3_distill_only/c3a_t12_historical_heads/seed_1337/manifest.v1.json \
  exp/chapter3_distill_only/c3b_t12_layer_to_layer/seed_1337/manifest.v1.json \
  --json-output exp/chapter3_distill_only/selection/c_report.json \
  --markdown-output exp/chapter3_distill_only/selection/c_report.md
```

## C-to-D gate and Conformer matching

After the scientific discussion identifies the C candidate, profile the
selected Transformer and a grid of Conformer FFN multiples of 128 with
`profile_config.py`. Then freeze the deterministic match:

```bash
scripts/distill_only/profile_config.py \
  --config-name d1_selected_transformer \
  --source-manifest PATH_TO_AGREED_C_MANIFEST \
  --destination-arch transformer \
  --output exp/chapter3_distill_only/selection/d1_profile.json

scripts/distill_only/profile_config.py \
  --config-name d2_selected_conformer \
  --source-manifest PATH_TO_AGREED_C_MANIFEST \
  --destination-arch conformer \
  --override model.student_ffn_dim=1536 \
  --output exp/chapter3_distill_only/selection/c6_ffn1536.json

scripts/distill_only/match_conformer.py \
  --target exp/chapter3_distill_only/selection/d1_profile.json \
  --candidate 1280=exp/chapter3_distill_only/selection/c6_ffn1280.json \
  --candidate 1408=exp/chapter3_distill_only/selection/c6_ffn1408.json \
  --candidate 1536=exp/chapter3_distill_only/selection/c6_ffn1536.json \
  --output exp/chapter3_distill_only/selection/conformer_match.json

scripts/distill_only/freeze_selection.py \
  --kind c_to_d \
  --output exp/chapter3_distill_only/selection/c_to_d.json \
  --candidate PATH_TO_C1_MANIFEST \
  --candidate PATH_TO_C2_MANIFEST \
  --candidate PATH_TO_C3A_MANIFEST \
  --candidate PATH_TO_C3B_MANIFEST \
  --selected PATH_TO_AGREED_C_MANIFEST \
  --metric validation_babble_0db_wer \
  --rationale 'record the agreed scientific rationale' \
  --approver 'name' \
  --conformer-match exp/chapter3_distill_only/selection/conformer_match.json
```

The selection lock snapshots clean and babble-0-dB validation values and
artifact hashes. Each profile is bound to the selected C manifest's immutable
digest. The matcher rejects a CLI FFN that differs from the profiled config
and rejects changes other than Transformer-to-Conformer block type and the
candidate FFN. D2 refuses to launch if the match artifact or any input profile
changes, and the matched FFN override cannot be superseded manually.

```bash
scripts/distill_only/launch.sh \
  --experiment d2_selected_conformer --stage all \
  --from-selection exp/chapter3_distill_only/selection/c_to_d.json
```

D1 is an exact alias of the selected Transformer run.

## D-to-E, optional S2, and final test gate

Create `selection/d_to_e.json` with the same selection tool after the D
validation discussion. E1 reuses the selected clean run; E2 materializes its
full resolved protocol and is allowed to change only train-time distillation
noise:

```bash
scripts/distill_only/launch.sh \
  --experiment e2_selected_noisy --stage all \
  --from-selection exp/chapter3_distill_only/selection/d_to_e.json
```

After validation review, create `selection/final.json` with
`--kind final`. This transitions exactly the selected run to
`selection_frozen`; test decoding rejects every other manifest. S2 is only an
optional diagnostic:

```bash
scripts/distill_only/launch.sh \
  --experiment s2_optional_two_stage --stage all \
  --from-selection exp/chapter3_distill_only/selection/final.json
```

Stage 1 is ordinary 50k distillation. Stage 2 loads the stage-1 student and
heads with `initialization_policy=warm_start_distilled`, then starts a fresh
25k optimizer/scheduler. Interruption resume inside either stage uses that
stage's own `checkpoint_last.pt`.

## Noise protocol

Training and reported evaluation deliberately use different mixers:

| Stage | Noise |
|---|---|
| Encoder distillation (clean B/C/D/E1) | none (`task.noise_prob=0.0`) |
| Encoder distillation (E2) | RMS, `p=0.25`, SNR `0 dB`, train-only |
| Downstream ASR fine-tuning | RMS, `p=0.25`, SNR `0 dB` (same overrides as DP: wav/prob/snr) |
| Screening validation (`--stage validate`) | ITU-T P.56 |
| Final inference (`--stage test`) | ITU-T P.56 |

The versioned protocol file is
`scripts/distill_only/evaluation_protocol_itut.yaml`
(`schema_version: chapter3-evaluation/v1`). It is loaded by the launcher
through `--evaluation-protocol` and recorded (path, SHA256, parsed matrix,
evaluation seed, speech level `-26 dBov`, noise-root hashes) in every run
manifest. Do not put evaluation-only fields into the Fairseq training
Hydra schema.

### Screening validation

`--stage validate` decodes LRS3 `valid` only:

| Condition | Method |
|---|---|
| `clean` | none |
| `babble_0db` | ITU-T, babble, 0 dB |
| `speech_0db` | ITU-T, speech, 0 dB |

Selection locks continue to read only `clean` and `babble_0db`. Adding
`speech_0db` does not change C-to-D or D-to-E selection rules.

### Final evaluation

After `selection/final.json` freezes the selected run to `selection_frozen`,
`--stage test` expands the full matrix on both `valid` and `test`:

- clean
- babble / music / speech
- SNRs `-10`, `-5`, `0`, `5`, `10` dB

That is 16 conditions per subset and 32 decode commands in total. Stable
condition names use `m` / `p` for signed SNRs (`babble_m10db`,
`music_p5db`, …). Every noisy command forces:

```text
override.noise_prob=1
override.noise_method=itut
override.noise_snr=<snr>
common.seed=<evaluation-seed>
```

The evaluation seed defaults to the protocol value `1337` and is independent
of the model-training `--seed` so every model shares the same noise-selection
protocol. Clean commands omit all `override.noise_*` keys.

Results are stored under distinct trees so RMS and ITU-T never collide and
screening never overwrites final artifacts:

```text
evaluation/screening/itut/{clean|babble|speech}/...
evaluation/final/itut/{clean|babble|music|speech}/...
```

### Example dry runs

```bash
scripts/distill_only/launch.sh \
  --experiment c2_t6_historical_heads \
  --stage validate \
  --evaluation-protocol scripts/distill_only/evaluation_protocol_itut.yaml \
  --dry-run
```

After final selection:

```bash
scripts/distill_only/launch.sh \
  --experiment s1_selected_main \
  --stage test \
  --from-selection exp/chapter3_distill_only/selection/final.json \
  --evaluation-protocol scripts/distill_only/evaluation_protocol_itut.yaml \
  --dry-run
```

ITU-T decoding requires the sibling package `itut_p56_noise_addition` (located
by `avhubert/noise_utils.py`) and the noise manifests:

- `/beegfs/data/shared/lrs3/noise/musan/tsv/babble/{valid,test}.tsv`
- `/beegfs/data/shared/lrs3/noise/musan/tsv/music/{valid,test}.tsv`
- `/beegfs/data/shared/lrs3/noise/speech/{valid,test}.tsv`

Missing ITU-T support or manifests fails before the first decode command.
