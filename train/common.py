"""Shared training utilities; all checkpoints are written outside tracked source."""

from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

def add_args(parser, default_batch):
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=default_batch)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--val-crop-size", type=int, default=128, help="0 uses full validation images")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--nf", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints")
    parser.add_argument("--resume", type=Path)
    return parser


def device_from_args(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    return torch.device(name)


def loader(dataset, args, shuffle, device):
    return DataLoader(dataset, batch_size=args.batch_size if shuffle else 1, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=device.type == "cuda")


def checkpoint_state(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint.get("model_state_dict", checkpoint)


def rgb_to_ycbcr(rgb):
    # Preserve the legacy TCPDNet loss scale used by the original training code.
    r, g, b = rgb.unbind(dim=2)
    return torch.stack((65.481*r + 128.553*g + 24.966*b + 16,
                        -37.797*r - 74.203*g + 112*b + 128,
                        112*r - 93.786*g - 18.214*b + 128), dim=2)


def demosaicking_loss(outputs, colors, labels):
    # Sub-Bayer order is 90°, 45°, 135°, 0° at TL, TR, BL, BR.
    color_labels = torch.stack((labels[:, 2, :, 0::2, 0::2],
                                labels[:, 1, :, 0::2, 1::2],
                                labels[:, 3, :, 1::2, 0::2],
                                labels[:, 0, :, 1::2, 1::2]), dim=1)
    return F.l1_loss(colors, color_labels) + 4 * F.l1_loss(rgb_to_ycbcr(outputs), rgb_to_ycbcr(labels))


def fit(model, optimizer, train_data, valid_data, loss_fn, args, filename):
    device = device_from_args(args.device)
    model.to(device)
    train_loader = loader(train_data, args, True, device)
    valid_loader = loader(valid_data, args, False, device)
    start_epoch, best_loss = 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch, best_loss = state["epoch"], state["best_val_loss"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_total = 0.0
        for batch in train_loader:
            batch = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch)
            loss.backward()
            optimizer.step()
            train_total += loss.item()
        model.eval()
        valid_total = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                batch = [item.to(device) for item in batch]
                valid_total += loss_fn(model, batch).item()
        valid_loss = valid_total / len(valid_loader)
        print(f"epoch {epoch+1}/{args.epochs}: train={train_total/len(train_loader):.6f} valid={valid_loss:.6f}", flush=True)
        best = valid_loss < best_loss
        best_loss = min(best_loss, valid_loss)
        state = {"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "best_val_loss": best_loss,
                 "nf": args.nf}
        torch.save(state, args.output_dir / filename)
        if best:
            torch.save(state, args.output_dir / filename.replace(".pth", "_best.pth"))
