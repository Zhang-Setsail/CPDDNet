"""Paper Stage II: freeze the DN/DM backbones and train the five GFMs."""

import argparse
from pathlib import Path

import torch

from common import add_args, checkpoint_state, demosaicking_loss, fit
from data import NoisyRawDataset
from models import PIDNDM


def load_stage1(model, dn_path, dm_path):
    model.dn_model.load_state_dict(checkpoint_state(dn_path))
    dm_state = checkpoint_state(dm_path)
    color_state = {key.removeprefix("color_unet."): value for key, value in dm_state.items()
                   if key.startswith("color_unet.")}
    polar_state = {key.removeprefix("polar_unet."): value for key, value in dm_state.items()
                   if key.startswith("polar_unet.")}
    if not color_state or not polar_state:
        raise ValueError("Stage I DM checkpoint must include color_unet and polar_unet")
    model.polar_unet.load_state_dict(polar_state)
    fusion_state = model.color_unet.state_dict()
    for key in fusion_state:
        if key.startswith("fusion_block_"):
            continue
        source_key = key.replace("shared_up", "up")
        if source_key not in color_state or fusion_state[key].shape != color_state[source_key].shape:
            raise ValueError(f"Cannot initialize fusion backbone parameter {key}")
        fusion_state[key] = color_state[source_key]
    model.color_unet.load_state_dict(fusion_state)
    # Start from the Stage I backbone's output while retaining gradients through
    # the final GFM projection; zeroing every GFM parameter would block learning.
    for name, module in model.color_unet.named_children():
        if name.startswith("fusion_block_"):
            torch.nn.init.zeros_(module.pointwise_conv2.weight)
            torch.nn.init.zeros_(module.pointwise_conv2.bias)


def freeze_backbones(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gfm_parameters = []
    for name, parameter in model.color_unet.named_parameters():
        if name.startswith("fusion_block_"):
            parameter.requires_grad_(True)
            gfm_parameters.append(parameter)
    return gfm_parameters


def loss_fn(model, batch):
    noisy, _, labels = batch
    outputs, colors = model(noisy)
    return demosaicking_loss(outputs, colors, labels)


def main():
    parser = add_args(argparse.ArgumentParser(description=__doc__), default_batch=1)
    default_dir = Path(__file__).resolve().parent / "checkpoints"
    parser.add_argument("--dn-checkpoint", type=Path, default=default_dir / "stage1_dn_best.pth")
    parser.add_argument("--dm-checkpoint", type=Path, default=default_dir / "stage1_dm_best.pth")
    args = parser.parse_args()
    model = PIDNDM(nf=args.nf)
    if args.resume is None:
        load_stage1(model, args.dn_checkpoint, args.dm_checkpoint)
    optimizer = torch.optim.Adam(freeze_backbones(model), lr=args.lr)
    fit(model, optimizer, NoisyRawDataset("train", args.crop_size),
        NoisyRawDataset("valid", args.val_crop_size or None), loss_fn, args, "stage2_cpddnet.pth")


if __name__ == "__main__":
    main()
