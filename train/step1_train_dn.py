"""Paper Stage I (part 1): train RAW denoising with an L1 loss."""

import argparse

import torch
from torch.nn import functional as F

from common import add_args, fit
from data import NoisyRawDataset
from models import SubModelDn


def loss_fn(model, batch):
    noisy, clean = batch
    return F.l1_loss(model(noisy), clean)


def main():
    args = add_args(argparse.ArgumentParser(description=__doc__), default_batch=16).parse_args()
    model = SubModelDn({"in_nc": 16, "out_nc": 16, "nf": args.nf})
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    fit(model, optimizer, NoisyRawDataset("train", args.crop_size, include_labels=False),
        NoisyRawDataset("valid", args.val_crop_size or None, include_labels=False),
        loss_fn, args, "stage1_dn.pth")


if __name__ == "__main__":
    main()
