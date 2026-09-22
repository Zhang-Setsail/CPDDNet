"""Paper Stage I (part 2): train color/polarization demosaicking on clean RAW."""

import argparse
from pathlib import Path

import torch

from common import add_args, checkpoint_state, demosaicking_loss, fit
from data import CleanRawDataset
from models import TCPDNet
from models.pidndm_model import unpack_raw


def loss_fn(model, batch):
    clean, labels = batch
    outputs, colors = model(unpack_raw(clean))
    return demosaicking_loss(outputs, colors, labels)


def main():
    parser = add_args(argparse.ArgumentParser(description=__doc__), default_batch=1)
    parser.add_argument("--color-init", type=Path, help="optional color UNet state dict")
    parser.add_argument("--polar-init", type=Path, help="optional polar UNet state dict")
    args = parser.parse_args()
    model = TCPDNet(nf=args.nf)
    if args.color_init:
        model.color_unet.load_state_dict(checkpoint_state(args.color_init))
    if args.polar_init:
        model.polar_unet.load_state_dict(checkpoint_state(args.polar_init))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    fit(model, optimizer, CleanRawDataset("train", args.crop_size),
        CleanRawDataset("valid", args.val_crop_size or None), loss_fn, args, "stage1_dm.pth")


if __name__ == "__main__":
    main()
