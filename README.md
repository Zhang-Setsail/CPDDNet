# CPDDNet: Color-Polarization Denoising and Demosaicking Network

Official repository for **CPDDNet**, a joint color-polarization denoising and demosaicking network for CPFA sensors.

Paper: [ICIP2026_Zhang.pdf](assets/ICIP2026_Zhang.pdf)

<p align="center">
  <img src="assets/overview.png" width="760" alt="Overview of CPDDNet and baseline pipelines">
</p>

## Introduction

Color-polarization filter array (CPFA) sensors capture color texture and polarization information in a single shot, but the raw mosaic data suffer from both noise and missing samples. Existing pipelines usually solve denoising (DN) and demosaicking (DM) separately, which can propagate noise or oversmooth structures needed for polarization recovery.

CPDDNet follows a DN-to-DM design and introduces a feature fusion module to retain raw CPFA information through both stages. This improves reconstructed color-polarization images and polarization parameters under severe noise.

<p align="center">
  <img src="assets/pipeline_pd.png" width="760" alt="CPDDNet pipeline and data structure">
</p>

## Release Plan

- [x] **Demo inference**: currently available. The `demo/` folder provides inference scripts, demo data, and instructions for running the pretrained models.
- [ ] **Evaluation results**: the `eval/` folder with released results and metric computation code will be released later. It may be absent from the first public GitHub release.
- [ ] **Training code and training dataset**: the full training pipeline and dataset preparation will be released after the evaluation package.

<!--
## Environment

Python 3.9+ is recommended.

```bash
conda create -n cpddnet python=3.10
conda activate cpddnet
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install PyTorch from the official PyTorch instructions first, then install the remaining packages from `requirements.txt`.
-->

## Demo Usage

First unzip the demo dataset:

```bash
cd demo
unzip data/FinalDataHigh.zip -d data/
```

Download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1Ca-RAS9Vu6enpmZ1bqcF_OMI6MLk0x36?usp=drive_link), then place them under:

```text
demo/pretrainedModel/
```

The demo scripts expect the pretrained model files to be stored in this directory.

Run CPDDNet inference:

```bash
python demo_ours.py
```

The outputs will be written to:

```text
demo/results/
```

Additional baseline demo scripts are also provided:

```bash
python demo_dm.py
python demo_dndm.py
python demo_dmdn.py
```

## Results

CPDDNet outperforms existing DM-only, DM-to-DN, and DN-to-DM baselines on the high-noise color-polarization dataset used in the paper.

<p align="center">
  <img src="assets/comparison_result_hq.png" width="760" alt="Visual comparison">
</p>

## Acknowledgement

The evaluation code is modified from [PolarDenDem](https://github.com/ymonno/EARI-Polarization-Demosaicking). The training code and dataset will be released in a later stage.
