# Training CPDDNet from scratch

This guide starts with a fresh checkout and ends with a Stage II CPDDNet checkpoint. Run every command from the **repository root** (`CPDDNet/`), unless a command explicitly changes directory. The required training datasets are available as ZIP archives in the same [Google Drive folder](https://drive.google.com/drive/folders/1Ca-RAS9Vu6enpmZ1bqcF_OMI6MLk0x36?usp=drive_link) as the pretrained model weights; they are not stored in this Git repository.

## What the three programs train

| Program | Paper stage | Input and target | What is updated | Required checkpoint |
| --- | --- | --- | --- | --- |
| `train/step1_train_dn.py` | Stage I, DN | Noisy CPFA RAW → clean CPFA RAW | RAW denoising network | None |
| `train/step2_train_dm.py` | Stage I, DM | Clean CPFA RAW → four RGB polarization images | Color and polarization demosaicking networks | None |
| `train/step3_train_gfm.py` | Stage II | Noisy CPFA RAW and its denoised version → four RGB polarization images | Five gated fusion modules (GFMs) only | Stage I DN **and** DM checkpoints |

The DN loss is L1. Both DM steps use color L1 plus four times polarization YCbCr L1, following the original training implementation. Stage II freezes the DN network and the Stage I DM backbones. The output of Step 3 is the complete CPDDNet model.

## 1. Set up Python

You need Python 3.9 or newer, `pip`, enough free space for the datasets and checkpoints, and the dependencies in [`requirements.txt`](../requirements.txt): PyTorch, NumPy, and OpenCV. A GPU is optional; the programs also run on CPU. `--device auto` selects CUDA, then Apple MPS, then CPU according to availability.

On macOS or Linux:

```bash
git clone https://github.com/Zhang-Setsail/CPDDNet.git
cd CPDDNet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, use `python -m venv .venv` and `.venv\Scripts\Activate.ps1` in place of the two environment commands above. If you intend to use CUDA, install a PyTorch build compatible with your CUDA environment before installing the remaining dependencies.

Check that the training entry points can import their dependencies:

```bash
python train/step1_train_dn.py --help
python train/step2_train_dm.py --help
python train/step3_train_gfm.py --help
```

These commands only display the available arguments; they do not start training.

## 2. Prepare the data

Download **`FinalDataHigh.zip`** and **`FinalDmData.zip`** from the shared [Google Drive folder](https://drive.google.com/drive/folders/1Ca-RAS9Vu6enpmZ1bqcF_OMI6MLk0x36?usp=drive_link). Both are required before training can start. The small dataset under `demo/data/` contains test examples and cannot replace them. The pretrained model weights in that folder are optional for Step 2 and are not needed for a fresh three-step training run.

Each archive already contains a top-level directory named `FinalDataHigh/` or `FinalDmData/`. Extract **both ZIPs into `dataset/`**, not into an additional directory with the same name. On macOS or Linux, from the repository root:

```bash
mkdir -p dataset
unzip "/path/to/FinalDataHigh.zip" -d dataset/
unzip "/path/to/FinalDmData.zip" -d dataset/
```

On Windows PowerShell:

```powershell
Expand-Archive -LiteralPath "C:\path\to\FinalDataHigh.zip" -DestinationPath dataset
Expand-Archive -LiteralPath "C:\path\to\FinalDmData.zip" -DestinationPath dataset
```

Replace the example ZIP paths with the locations where your browser saved the downloads. After extraction, the directory layout must be:

```text
CPDDNet/
├── dataset/
│   ├── FinalDataHigh/
│   │   ├── input_train/       # Scene1_1.npy, ... (960 noisy RAW samples)
│   │   ├── input_valid/       # Scene31_1.npy, ... (5 noisy RAW samples)
│   │   └── label/
│   │       ├── Scene1.npy     # clean RAW; one per scene
│   │       ├── Scene1_0.png
│   │       ├── Scene1_45.png
│   │       ├── Scene1_90.png
│   │       └── Scene1_135.png
│   └── FinalDmData/
│       ├── train/             # Scene1.npy, ... (30 scene names)
│       ├── valid/             # Scene31.npy, ... (5 scene names)
│       └── label/             # Scene1_{0,45,90,135}.png, ...
└── train/
```

The tree shows representative filenames, not the complete contents. Each RAW `.npy` must contain a two-dimensional mosaic whose height and width are divisible by 4; use dimensions divisible by 64 if you plan to validate on full images, because the networks downsample several times. Each RGB label must match its scene's RAW image dimensions. The readers convert RAW to `float32` and RGB PNG values to the `[0, 1]` range; noisy RAW values are not clipped.

If you already have the extracted datasets elsewhere, you can link them instead of downloading and extracting the archives. On macOS or Linux, create local symbolic links from the repository root and replace the example source paths with the actual absolute paths on your machine:

```bash
mkdir -p dataset
ln -s "/absolute/path/to/FinalDataHigh" dataset/FinalDataHigh
ln -s "/absolute/path/to/FinalDmData" dataset/FinalDmData
```

On Windows PowerShell, directory junctions serve the same purpose:

```powershell
New-Item -ItemType Junction -Path dataset\FinalDataHigh -Target "D:\data\FinalDataHigh"
New-Item -ItemType Junction -Path dataset\FinalDmData -Target "D:\data\FinalDmData"
```

Choose either extraction or links for each dataset; do not extract an archive onto an existing dataset link. Extracted data and local dataset links are ignored by Git. `HighRGB2RGB` is **not** needed for these three training steps.

Step 1 reads noisy RAW from `FinalDataHigh/input_train` and `input_valid` and clean RAW from `FinalDataHigh/label`; it does not need RGB labels. Step 3 reads noisy RAW and four RGB labels from `FinalDataHigh` (its data reader also checks the aligned clean RAW). Step 2 takes the training/validation **scene names** and RGB labels from `FinalDmData`, but reads their clean RAW from `FinalDataHigh/label`. It does not read the pixel contents of `FinalDmData/train/*.npy` or `valid/*.npy`. `input_test` and `test` are not needed for training.

Before starting, check that both directories are reachable:

```bash
ls dataset/FinalDataHigh/input_train
ls dataset/FinalDataHigh/input_valid
ls dataset/FinalDataHigh/label
ls dataset/FinalDmData/train
ls dataset/FinalDmData/valid
ls dataset/FinalDmData/label
```

## 3. Train each step

The three commands below can be run in sequence. Step 1 and Step 2 are independent and may also be run separately. Step 3 needs the checkpoints they produce. A fresh training run does **not** need any files from `pretrainedModel/`.

### Step 1 — Stage I RAW denoising

```bash
python train/step1_train_dn.py --device auto --epochs 200 --batch-size 16
```

This trains on 960 noisy RAW images from 30 scenes and validates on 5 scenes. It saves:

```text
train/checkpoints/stage1_dn.pth       # latest epoch
train/checkpoints/stage1_dn_best.pth  # lowest validation loss so far
```

To run only this step on a particular device or with a smaller batch, for example:

```bash
python train/step1_train_dn.py --device mps --batch-size 2 --output-dir /path/to/dn-checkpoints
```

Use `--device cuda`, `mps`, or `cpu` only when that device is available. If you choose a custom output directory, pass its resulting checkpoint path explicitly to Step 3.

### Step 2 — Stage I clean RAW demosaicking

```bash
python train/step2_train_dm.py --device auto --epochs 200 --batch-size 1
```

This trains the color and polarization UNets together from scratch on 30 clean RAW scenes and validates on 5 scenes. It saves:

```text
train/checkpoints/stage1_dm.pth       # latest epoch
train/checkpoints/stage1_dm_best.pth  # lowest validation loss so far
```

If you already have compatible **individual UNet state dictionaries**, you can optionally initialize this step with them. Both options may be used independently; omit them for training from scratch:

```bash
python train/step2_train_dm.py \
  --color-init /path/to/color_unet.pth \
  --polar-init /path/to/polar_unet.pth
```

These optional files initialize Step 2 only. Step 3 needs the combined `stage1_dm_best.pth` (or another checkpoint produced by Step 2), not two standalone UNet files.

### Step 3 — Stage II gated fusion

After Steps 1 and 2 have saved their best checkpoints:

```bash
python train/step3_train_gfm.py --device auto --epochs 200 --batch-size 1
```

By default, Step 3 loads `train/checkpoints/stage1_dn_best.pth` and `train/checkpoints/stage1_dm_best.pth`. It initializes the fusion color backbone from Stage I, freezes the DN and DM backbones, and updates only the five GFMs. It saves:

```text
train/checkpoints/stage2_cpddnet.pth       # latest epoch; complete model
train/checkpoints/stage2_cpddnet_best.pth  # lowest validation loss; complete model
```

If the Stage I checkpoints are elsewhere, specify **both** paths:

```bash
python train/step3_train_gfm.py \
  --dn-checkpoint /path/to/stage1_dn_best.pth \
  --dm-checkpoint /path/to/stage1_dm_best.pth \
  --output-dir /path/to/cpddnet-checkpoints
```

The three steps must use the same `--nf` value; the default is 64. Checkpoint loading fails if the model widths differ. Step 3 can be started on its own only if suitable Stage I DN and DM checkpoints already exist.

## Training options and resuming

All three programs accept these common options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--epochs` | `200` | Total number of epochs, including epochs already completed when resuming. |
| `--batch-size` | `16` for Step 1; `1` for Steps 2 and 3 | Training batch size. Validation always uses batch size 1. |
| `--crop-size` | `128` | Random training RAW crop size, aligned to the 4×4 CPFA pattern. Use a multiple of 64, such as 64 or 128, so every down/up sampling level has matching dimensions. |
| `--val-crop-size` | `128` | Centered validation RAW crop size. Use a multiple of 64, or set to `0` for full images (which need more memory). |
| `--lr` | `1e-4` | Adam learning rate. |
| `--nf` | `64` | Base model width; keep it consistent across the three steps and their checkpoints. |
| `--num-workers` | `0` | Number of data loader worker processes. |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu`. |
| `--output-dir` | `train/checkpoints/` | Directory for the latest and best checkpoints. |
| `--resume` | None | Resume the **same step** from a checkpoint including optimizer state. |

For example, to continue Step 1 from the latest checkpoint until a **total** of 250 epochs:

```bash
python train/step1_train_dn.py \
  --resume train/checkpoints/stage1_dn.pth \
  --epochs 250
```

Similarly, resume Step 2 with `step2_train_dm.py --resume train/checkpoints/stage1_dm.pth --epochs 250`, or Step 3 with `step3_train_gfm.py --resume train/checkpoints/stage2_cpddnet.pth --epochs 250`. When resuming Step 3, `--resume` supplies the entire model and optimizer state, so the Stage I paths are not needed. Use the same `--nf` and training settings as the original run. If the original run used `--output-dir`, set it again so new checkpoints are written there.

Each completed epoch prints mean training and validation loss. The `*_best.pth` checkpoint is selected using the configured validation crop, so changing `--val-crop-size` changes the basis for that comparison. Checkpoints and locally linked datasets are ignored by Git; no test or generated training files need to be added to the source tree.

## Common setup errors

- **No `.npy` files found:** verify that both archives were extracted into `dataset/` (or that the dataset links resolve), and check the exact `input_train`, `input_valid`, `train`, and `valid` directory names. Avoid an extra `FinalDataHigh/FinalDataHigh` or `FinalDmData/FinalDmData` directory layer.
- **Missing label or mismatched shapes:** each scene needs its clean RAW and four `{0,45,90,135}` RGB PNG labels at matching dimensions. Check the scene name before the underscore in a noisy RAW filename.
- **MPS/CUDA unavailable:** use `--device auto` or `--device cpu`, or install a PyTorch build that supports your hardware.
- **Out of memory:** reduce `--batch-size`; keep `--val-crop-size` at 128 instead of 0. Use 64 as the smallest RAW crop for all three steps.
- **Step 3 cannot find a checkpoint:** finish Steps 1 and 2 first, or provide both `--dn-checkpoint` and `--dm-checkpoint` paths.
