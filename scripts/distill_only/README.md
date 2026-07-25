# Chapter 3 distillation-only execution

All commands below run from the repository root on
`feat/ch3-distill-only`. The launcher requires a fully clean worktree and
verifies the fixed branch ancestor, teacher, LRS3 manifests, tokenizer, and
MUSAN training manifest by SHA256 before it creates a run.

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
artifact hashes. D2 refuses to launch if the match artifact or any input
profile changes, and the matched FFN override cannot be superseded manually.

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
