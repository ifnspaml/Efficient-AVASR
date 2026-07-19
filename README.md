# Efficient Noise-Robust Hybrid Audiovisual Encoder with Joint Distillation and Pruning for Audiovisual Speech Recognition

This is the official implementation of our Interspeech 2025 paper: [Efficient Noise-Robust Hybrid Audiovisual Encoder with Joint Distillation and Pruning for Audiovisual Speech Recognition](https://www.isca-archive.org/interspeech_2025/li25q_interspeech.html).

## Installation
First, create a conda virtual environment and activate it:
```
conda create -n dpavhubert python=3.8 -y
conda activate dpavhubert
```
Then, clone this directory:
```
git clone https://github.com/Cyion/dpav_hubert.git
cd dpav_hubert
git submodule init
git submodule update
```

Lastly, install fairseq and the other packages:
```
pip install -r requirements.txt
cd fairseq
pip install --editable ./
```

## Joint Distillation and Pruning
To run a joint distillation and pruning experiment execute the following script:
```
sbatch scripts/run_pruning.sh
```
For configuration see scripts/run_pruning.sh. Configuration files can be found in avhubert/conf/.

## Joint Distillation and Merging
To run a joint distillation and merging experiment execute the following script:
```
sbatch scripts/run_merging.sh
```
For configuration see scripts/run_merging.sh. Configuration files can be found in avhubert/conf/.

## Joint Distillation, Pruning and Merging
To run a joint distillation, pruning and merging experiment execute the following script:
```
sbatch scripts/run_pruning_merging.sh
```
For configuration see scripts/run_pruning_merging.sh. Configuration files can be found in avhubert/conf/.

## Teacher Baseline (Full AV-HuBERT Fine-tuning)
To fine-tune the full teacher model (base encoder + S2S decoder) on LRS3 433h and run inference, submit to the GPU cluster:
```
sbatch scripts/run_baseline.sh
```
For configuration see scripts/run_baseline.sh. Outputs are written to `exp/finetune/asr/teacher/`.

Noisy inference uses RMS noise addition by default. For ITU-T P.56 active speech level mixing, set `INFER_NOISE_METHOD=itut` (or `p56`) when submitting; results are written under `infer_itut/` instead of `infer/`. Requires the sibling package `itut_p56_noise_addition` under `work_fast/`.

## Evaluate Model FLOPs
To evaluate the FLOPs of a model run:
```
python avhubert/eval_flops.py --ckpt <path to model checkpoint>
```

## Implementation Details

The teacher-student architecture (AVHubertDistill) is implemented in hubert_distill.py (for configuration see AVHubertDistillConfig in same file). The joint distillation and pruning, and joint distillation and merging objectives (AVHubertDistillCriterion) are implemented in hubert_distill_criterion.py (for configuration see AVHubertDistillCriterionConfig in same file). Removing the hard concrete masks from the model and pruning the components according to the mask values is implemented in prune.py. The avhubert model (see hubert.py) implements a prune-function, which prunes the components and returns the new config of the components. The hard concrete mask layer (HardConcrete) is implemented in hardconcrete.py and used in encoder.py and resnet.py to prune attention heads, FFN intermediate dimensions and CNN channels. The expected number of parameters of the model is calculated using the get_num_params-fumction. Pruning of the avhubert components, e.g. linear layers, conv2d layers, layer norms, and PReLU, is implemented in pruning_utils.py. Merging the query, key, and value projection weights is implemented in merge.py. The MHA layer has a merge-function which is called to merge the weights, where the parameter "t" corresponds to beta and "type" defines the projections to merge, e.g. QK, QV, KV, or QKV.

## Citation

If you find this work helpful, please cite the paper:

```bibtex
@inproceedings{li25q_interspeech,
  title     = {{Efficient Noise-Robust Hybrid Audiovisual Encoder  with Joint Distillation and Pruning for Audiovisual Speech Recognition}},
  author    = {{Zhengyang Li and Pascal Reichert and Thomas Graave and Patrick Blumenberg and Tim Fingscheidt}},
  year      = {{2025}},
  booktitle = {{Interspeech 2025}},
  pages     = {{1833--1837}},
  doi       = {{10.21437/Interspeech.2025-1464}},
  issn      = {{2958-1796}},
}
```
