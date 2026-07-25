# Chapter 3 distillation-only provenance

This document records the read-only audit used to implement the isolated
Chapter 3 distillation-only experiment suite. It distinguishes Git-tracked
history from ignored experiment artifacts and from the dirty state of the old
checkout.

## Repository baselines

### Unified repository

- Path: `/home/zhengyangli/work_fast/dpav_hubert_new`
- Remote: `git@github.com:ifnspaml/Efficient-AVASR.git`
- Required baseline branch: `dev_li_pro6000`
- Recorded baseline commit:
  `4502130ce4470fe8b0456fd5b466bef043f383f9`
- Implementation branch: `feat/ch3-distill-only`, created directly at that
  commit. The baseline was a clean tracked worktree before implementation.
- Joint-DP optimization reference:
  `avhubert/conf/distill/resnet.yaml` at the baseline commit, SHA256
  `913ba9fad29ed32edf964312053d3927f6a62c41006f36662edcd85c84f7d5de`.

The reference establishes the model peak LR `0.002`, Adam betas
`(0.9,0.999)`, epsilon `1e-8`, zero weight decay, polynomial power `1`,
15,000 warm-up updates, clipping at `10`, and the LRS3 data/batch convention.
Its composite optimizer and all pruning-specific settings are deliberately not
copied into distillation-only configs.

### Old distillation-only repository

- Path: `/home/zhengyangli/work/distil-av-hubert`
- Remote: `git@github.com:Chenwei-Liang/distil-av-hubert.git`
- Branch and HEAD observed during the audit: `master` at
  `13e27bcb44710c244ef48899d4458107fc35b2cb`
- The old checkout was dirty. The exact `git status --porcelain=v1` entries
  observed were:

  ```text
   M s3prl/hub.py
   M s3prl/model_summary.py
   M s3prl/pretrain/distiller/config_model.yaml
   M s3prl/pretrain/distiller/config_runner.yaml
   M s3prl/run_pretrain.py
   M s3prl/upstream/__init__.py
   M s3prl/upstream/avhubert/hubconf.py
   M s3prl/upstream/distiller/model.py
   M s3prl/upstream/distiller/resnet.py
  ?? s3prl/environment.yaml
  ```

- SHA256 of `git diff --binary HEAD`:
  `b5870684ec856a5ea24251428e74d30834ddaa533712a6e6ba24981b0335fc9a`.
- SHA256 of the status text above:
  `c89e34c979c59bcf6672e7d6a4522ef2bfa85db2c4a7fb803c82afa9151b75b4`.
- The untracked `s3prl/environment.yaml` is not represented in the Git diff;
  its SHA256 was
  `7e3e1eaf027d5177a021d9ff2ac18339f6cadb95c5cbb095c6ab8c387d661159`.

Consequently, tracked source behavior cited below is anchored to the
`13e27bc...` commit unless a filesystem-only archive is explicitly named.
Files below `s3prl/result/` are ignored by `.gitignore`; their content cannot
be attributed to that commit and is identified by content hash instead.

## Exact historical source map

### Prediction heads

- `s3prl/upstream/distiller/model.py`
  - `DistillerConfig` defines `out_layer_type`, `out_layer_inter_dim`,
    `task_emb_type`, `n_tasks`, and `pred_layer_id`.
  - `DistillerModel.__init__`, in the `expand-last` branches, constructs
    `Linear(final_student_dim, number_of_targets * hidden_dim)`, `GELU`, then
    `SplitLinear(hidden_dim, number_of_targets, teacher_dim)`.
  - `DistillerModel.forward` applies this stack only to the final student
    representation and reshapes it to `B x N x T x D`.
- `s3prl/upstream/distiller/module.py`
  - `SplitLinear` owns a distinct weight tensor of shape
    `N x hidden_dim x teacher_dim` and a distinct bias for every target.
    Its `einsum` applies the projections independently.

The historical implementation is target-count agnostic. Archived runs used
one, two, or three selected targets; using four heads for `[0,4,8,12]` is the
new controlled protocol, not a claim that an old four-target archive exists.

### Teacher-target selection

- `s3prl/upstream/avhubert/expert.py`, `UpstreamExpert.__init__`, registers a
  hook on every teacher encoder layer that captures the layer input and a final
  hook on the encoder output.
- `s3prl/upstream/interfaces.py`, `UpstreamBase.__call__`, preserves hook order
  as `hidden_states`.
- Therefore identifier `0` is the input to teacher block 1, identifiers
  `1..11` are the outputs of the preceding blocks, and identifier `12` is the
  final encoder output.
- `s3prl/pretrain/distiller/pretrain_expert.py`,
  `DistillerForPretrain.forward`, indexes `teacher_hiddens["hidden_states"]`
  with `pred_layer_id`, stacks the selected tensors as `B x N x T x D`, and
  compares them with the final-output head predictions.

This explains why teacher identifier `0` is valid while also explaining why it
must never be interpreted as student block 0 in historical-head mode.

### Distillation losses

`s3prl/pretrain/distiller/pretrain_expert.py` is the loss source:

- `DistillerForPretrain.__init__` selects elementwise L1 or L2 reconstruction.
- `compute_loss` asserts prediction/target shape equality and globally averages
  the elementwise reconstruction loss, which gives equal weighting to equal
  sized target tensors.
- Its cosine term is
  `-logsigmoid(cosine_similarity(prediction, target, dim=-1))`, globally
  averaged.
- Its optional feature penalty is the mean squared magnitude of the fused
  representation after the historical frontend `LayerNorm` and before the
  optional sequence-dimension projection.
- The historical total is reconstruction + configured feature penalty +
  configured cosine coefficient. The Chapter 3 defaults are L1 `1.0`, L2
  `0.0`, negative-log-sigmoid cosine `1.0`, and feature penalty `0.0`.

### Student architectures

The implementation itself is generic:

- `s3prl/upstream/distiller/model.py`, `DistillerConfig` and
  `DistillerModel`, select encoder type, depth, D, FFN width, heads, and
  frontend.
- `s3prl/upstream/distiller/module.py`, `TransformerEncoder` and
  `TransformerSentenceEncoderLayer`, define the Transformer stack.
- The following filesystem artifacts supply the historical dimension defaults:

| Student | Exact source artifact | SHA256 | Audited values |
|---|---|---|---|
| T2 | `s3prl/result/pretrain/DistilAVHuBERT_base_vox_av_resnet_transformer_encoder2-768-3072_targets4812_bs4_gradacc6_modalities_av_fp16/config_model.yaml` | `f2d866c17f7223dbbe1d4dccb44097563fa1b130f49eb60459713da6a8ea3583` | Transformer, 2 blocks, D=768, FFN=3072, 12 heads; archive targets `[4,8,12]` |
| T6 | `s3prl/pretrain/distiller/config_model.yaml` from Git commit `13e27bc...` | `71c64100f93322293791964a821e65a0b6253985a958979c1334ec4ac8c3f20a` | Transformer, 6 blocks, D=384, FFN=3200, 12 heads; tracked config targets `[8,12]` |
| T12 | `s3prl/result/pretrain/DistilAVHuBERT_learn_from_finetuned_av_resnet_transformer_encoder12-384-1024_bs4_gradacc6_modalities_av/config_model.yaml` | `b27f8905b446f3a39c3ddac8ee1a3f6ec82e696c1988518a86bae3fe456c11f9` | Transformer, 12 blocks, D=384, FFN=1024, 12 heads; archive targets `[4,8,12]` |
| C6 | `s3prl/result/pretrain/DistilAVHuBERT_learn_from_finetuned_av_resnet_conformer_encoder6-384-1536_bs4_gradacc6_modalities_av_fp16no/config_model.yaml` | `ceb6267f47bba201e5b1311346aaff287a99898c5f8231917356fb2eaf3f08f4a7` | Conformer, 6 blocks, D=384, FFN=1536, 12 heads, kernel 31; archive targets `[8,12]` |

There is no archived T6 Transformer result directory in the inspected
`s3prl/result/pretrain/` tree. Its architecture provenance is only the tracked
default config at commit `13e27bc...`; this gap must remain explicit in result
manifests and reports.

#### Compact T12 matching criterion

The compact T12 default remains D=384, FFN=1024, and 12 heads. A numeric
pre-launch profile on the required baseline, using audio `[1,104,100]`, video
`[1,1,100,88,88]`, the `ptflops` aten backend, and FLOPs defined as twice
MACs, produced:

| Student | Deployed parameters | Four-head parameters | Student + heads | MACs | FLOPs |
|---|---:|---:|---:|---:|---:|
| T2 D768/FFN3072 | 31,741,672 | 4,724,736 | 36,466,408 | 33,705,583,360 | 67,411,166,720 |
| T6 D384/FFN3200 | 31,226,344 | 1,774,080 | 33,000,424 | 33,665,871,232 | 67,331,742,464 |
| T12 D384/FFN1024 | 29,470,696 | 1,774,080 | 31,244,776 | 33,535,464,832 | 67,070,929,664 |
| Provisional C6 D384/FFN1536/K31 | 32,201,960 | 1,774,080 | 33,976,040 | 33,714,342,400 | 67,428,684,800 |

Thus T12 is not falsely described as exactly parameter matched: its deployed
backbone is smaller, but its total training model remains in the intended
approximately 31–32 M compact range, and its audiovisual FLOPs differ by less
than 0.51% from both depth controls. This joint criterion preserves the
audited T12 architecture while tightly matching compute. The machine-readable
record is `docs/chapter3_distill_only_architecture_profile.v1.json`; every
launched run profiles itself again and stores the numeric result in its
manifest.

### Conformer block

- Wrapper/config source:
  `s3prl/upstream/distiller/module.py`, `ConformerEncoder`, plus
  `s3prl/upstream/distiller/model.py`, `DistillerConfig`.
- The wrapper imports
  `fairseq.modules.conformer_layer.ConformerWav2Vec2EncoderLayer`.
- The old environment resolves that import to
  `/beegfs/work/zhengyangli/av-hubert/fairseq/fairseq/modules/conformer_layer.py`.
  That Fairseq checkout has remote
  `git@github.com:Chenwei-Liang/av-hubert.git`, commit
  `0ba0026f815bf41d162d490f16a702c5d41c1fa8`; the conformer source was clean
  and had SHA256
  `bc70f7245cee66f1efd2212089811d5b88cd7ec361b66b8cafdab1f1ce3a1e97`.
- `ConformerEncoderLayer` is a Macaron block: pre-normalized FFN1 with `0.5`
  residual scaling, pre-normalized self-attention, pre-normalized convolution,
  FFN2 with `0.5` residual scaling, then final LayerNorm.
- Empty historical `attn_type` takes the standard Fairseq
  `MultiheadAttention` branch. Historical `pos_enc_type` is `abs`; in the
  wrapper this creates no separate relative/rotary position tensor.
- `ConvolutionModule` uses LayerNorm, bias-free pointwise convolution, GLU,
  bias-free odd-kernel depthwise convolution, BatchNorm, Swish, bias-free
  pointwise projection, and dropout. The archived C6 kernel is 31.

### Initialization and frontend policy

`s3prl/pretrain/distiller/pretrain_expert.py`,
`DistillerForPretrain.__init__`, defines the historical initialization:

- `init_teacher_conv_layers` loads the teacher audio frontend state, video
  frontend state, and fusion/post-extraction projection when present.
- `freeze_feature_extractor` optionally freezes the copied audio/video
  frontends.
- `init_teacher_encoder_layers` loads the positional convolution and copies
  the first `encoder_layers` teacher blocks.

The old code uses strict `load_state_dict`; it does not provide a
shape-compatible partial-copy policy. Direct sequence copying is therefore
compatible with the T2/D768 Transformer but not with D384 students. The new
`random_sequence_teacher_frontend` policy records every copied/skipped key and
reason, and retains dimension-dependent projections as random when shapes
differ.

The fixed teacher checkpoint has `dropout_input=0.1`, but the historical
distiller did not apply an additional pre-encoder input dropout. The isolated
student therefore exposes `student_dropout_input` explicitly and sets its
historical default to `0.0`; encoder, attention, and activation dropouts remain
explicitly recorded at `0.1`.

The frontend modules themselves are built in
`s3prl/upstream/distiller/model.py`: `SubModel`, the ResNet/ShuffleNet
selection, audio/video projections, modality fusion, and
`post_extract_proj`. Archived initialization flags are not evidence of copied
weights by themselves; for example the audited T2/T12/C6 archives set both
copy flags false.

### Noise augmentation

- The actual training import is
  `from pretrain.distiller.dataset import OnlineWaveDataset` in
  `s3prl/pretrain/distiller/pretrain_expert.py`.
- The active implementation is therefore
  `s3prl/pretrain/distiller/dataset.py`, not the duplicate
  `s3prl/pretrain/distiller/hubert_dataset.py`.
- `OnlineWaveDataset` resolves `${noise_wav_path}/${sets}.tsv`, performs one
  Bernoulli draw per audio sample, samples noise entries with replacement,
  repeats short noise or truncates long noise from offset zero, scales by
  clean/noise RMS to the selected SNR, adds the waveforms, rescales to avoid
  int16 clipping, casts to int16, and then computes log-filterbank features.
- With multiple noises it truncates them to their common shortest length and
  averages them before mixing. Numeric SNR is fixed; a tuple requests an
  integer draw over an inclusive range.
- The archived noisy run uses MUSAN
  `/beegfs/data/shared/lrs3/noise/musan/tsv/all`, probability `0.25`, and SNR
  `0 dB`. Its exact model and runner hashes are respectively
  `08471dee768d00ba9259eba3fa7d31b5f4326f7614f1fb9157fc3f0f0a6b5ac9`
  and
  `c6fb2e83d8aad900ee41772f103fe6a5e26603fe0a68b0be9b6d24b72c0895ad`.

The old dataset will use a split-specific noise manifest for any constructed
split. The Chapter 3 task intentionally tightens this to train-only noise and
forces validation/test distillation data to remain clean.

### Checkpoint save and resume

- `s3prl/pretrain/runner.py`
  - `_get_upstream`, `_get_optimizer`, and `_get_scheduler` restore model,
    optimizer, and scheduler state.
  - `train` stores optimizer, scheduler, configuration, arguments, a `Step`
    value, and the distiller state in `states-*.ckpt`; it prunes older files by
    parsed filename step.
- `s3prl/run_pretrain.py`
  - `--past_exp` accepts a checkpoint or directory.
  - `--auto_resume` searches the experiment directory and selects the
    numerically latest `states-*.ckpt`.
  - Saved arguments/configuration overwrite most newly supplied values.

Historical limitations that must not be reproduced:

1. `global_step` is `pbar.n + 1`, but the checkpoint stores `Step: pbar.n`
   before `pbar.update(1)`. Resume therefore repeats the displayed update while
   loading optimizer/scheduler state from after that update.
2. AMP scaler, dataloader iterator/epoch, RNG states, partial accumulation
   state, and meters are not saved, so continuation is not exact.
3. Auto-resume trusts filename parsing and does not validate a config or code
   digest.
4. Saved arguments can silently override new command-line intent.

The unified Fairseq checkpoint lifecycle is retained instead, with exact
within-stage optimizer/scheduler/update restoration. S2 stage transfer is
separate: weights warm-start stage 2 while optimizer, scheduler, meters,
iterator, and update count are intentionally reset.

## Archived-config hash inventory

All paths below are under `s3prl/result/pretrain/` in the old checkout. This
inventory covers every archived `config_model.yaml` and `config_runner.yaml`
observed during the audit.

| Archived run directory | `config_model.yaml` SHA256 | `config_runner.yaml` SHA256 |
|---|---|---|
| `DistilAVHuBERT_base_vox_av_resnet_transformer_encoder2-768-3072_targets4812_bs4_gradacc6_modalities_av_fp16` | `f2d866c17f7223dbbe1d4dccb44097563fa1b130f49eb60459713da6a8ea3583` | `41578d1cba64a7651e5e59069c593d25d1cffb49a95e451f3f55578c3779f4a7` |
| `DistilAVHuBERT_learn_from_finetuned_av_resnet_conformer_encoder6-384-1536_bs4_gradacc6_modalities_av_fp16no` | `ceb6267f47bba201e5b1311346aaff287a99898c5f8231917356fb2eaf3f08f4` | `15dfa5e32ec2de8716c50ad9cdee641633dc1dec237be0e96483cfa3a1e0b708` |
| `DistilAVHuBERT_learn_from_finetuned_av_resnet_transformer_encoder12-256-1536_bs4_gradacc6_modalities_av` | `9842809ef8e1c930173082ac037fcbd7b9383582f4edcfde02cb099aebe3a3d6` | `79af11d2f14b90bbac49b8258a6625fdd53604d7e5e92a55d99f2b69b42b90e7` |
| `DistilAVHuBERT_learn_from_finetuned_av_resnet_transformer_encoder12-384-1024_bs4_gradacc6_modalities_av` | `b27f8905b446f3a39c3ddac8ee1a3f6ec82e696c1988518a86bae3fe456c11f9` | `79af11d2f14b90bbac49b8258a6625fdd53604d7e5e92a55d99f2b69b42b90e7` |
| `DistilAVHuBERT_learn_from_finetuned_av_resnet_transformer_encoder12-384-1024_bs4_gradacc6_modalities_av_fp16no` | `b27f8905b446f3a39c3ddac8ee1a3f6ec82e696c1988518a86bae3fe456c11f9` | `15dfa5e32ec2de8716c50ad9cdee641633dc1dec237be0e96483cfa3a1e0b708` |
| `DistilAVHuBERT_learn_from_finetuned_av_resnet_transformer_encoder12-384-1024_targets812_bs4_gradacc6_modalities_av_fp16` | `21e9a7176233b9c9ecfcbe267e69ba4bdf27c3fac1fd03e12c1c7bb2b449315f` | `5f3407d5b5ec1c0dd30062027abf4a7701cc8829fa87612a54639c8f092d26f5` |
| `DistilAVHuBERT_learn_from_finetuned_av_resnet_transformer_encoder12-384-1024_targets812_bs4_gradacc6_modalities_av_fp16no` | `c2d58c0eaf4f5f439445c74caacfe550ee68694645ce73d4c4e8f86a7d2702f1` | `5f3407d5b5ec1c0dd30062027abf4a7701cc8829fa87612a54639c8f092d26f5` |
| `DistilAVHuBERT_learn_from_finetuned_av_resnet_transformer_encoder12-384-1024_targets8_bs4_gradacc6_modalities_av_fp16` | `21e9a7176233b9c9ecfcbe267e69ba4bdf27c3fac1fd03e12c1c7bb2b449315f` | `5f3407d5b5ec1c0dd30062027abf4a7701cc8829fa87612a54639c8f092d26f5` |
| `DistilAVHuBERT_learn_from_finetuned_av_resnet_transformer_encoder12-768-3072_targets812_bs4_gradacc6_modalities_av_fp16no` | `47c25ee70b3f951e9f87bb799402662fe1ffd63c11c63adc0b84d9f42dc716c1` | `db5a1c0b00e7aeba632b11eef81b80dd62e795d264f4b31f8d92f2ca9f0fb5a1` |
| `DistilAVHuBERT_learn_from_finetuned_av_shufflenet_conformer_encoder6-384-1536_bs4_gradacc6_modalities_av_fp16no` | `150d6b76449f0af341483acc0e5831c26a4e91b05e66a3fb886db1a512fb1254` | `15dfa5e32ec2de8716c50ad9cdee641633dc1dec237be0e96483cfa3a1e0b708` |
| `DistilAVHuBERT_learn_from_finetuned_av_shufflenetv2_512_conformer_encoder12-384-1536_targets812_bs4_gradacc6_modalities_av_fp16no` | `9e557baa243428d92c7d19d3f242d0deb95e62c47453b4510dc6fa7db91bd683` | `db5a1c0b00e7aeba632b11eef81b80dd62e795d264f4b31f8d92f2ca9f0fb5a1` |
| `DistilAVHuBERT_noiseAug_learn_from_finetuned_av_shufflenet_conformer_encoder6-384-1536_bs4_gradacc6_modalities_av_fp16no` | `08471dee768d00ba9259eba3fa7d31b5f4326f7614f1fb9157fc3f0f0a6b5ac9` | `c6fb2e83d8aad900ee41772f103fe6a5e26603fe0a68b0be9b6d24b72c0895ad` |

## New-suite interpretation

- Main teacher targets are always `[0,4,8,12]`.
- T2, T6, and T12 use the same historical final-output prediction-head
  formulation in the primary depth comparison.
- Only a student with every requested mapping may use layer-to-layer mode;
  the default direct mapping is student `[0,4,8,12]` to teacher
  `[0,4,8,12]`.
- Main configs use one continuous 75,000-update, pruning-free Adam schedule.
- D/E/S public configs contain an explicit provisional T6 materialization for
  dry-run validation, but launch-time selection manifests must replace it.
  Depth 6 is never an implicit fallback.
- S2 is an optional two-stage ordinary-distillation diagnostic: 50k at
  LR `0.002` with 15k warm-up, then a fresh 25k optimizer/scheduler stage at
  LR `0.0001` with 5k warm-up, warm-started from stage-1 model/head weights.
