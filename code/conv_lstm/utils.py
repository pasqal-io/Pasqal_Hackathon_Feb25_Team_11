import json
import os
import torch
from torch import nn, optim
import torch.nn.functional as F

from code.conv_lstm.losses import BinaryDiceLoss, FocalLoss
from code.conv_lstm.model import ConvLSTM


def load_json_config(path: str = "config.json"):
    with open(path, "r") as f:
        return json.load(f)


def create_padded_mask(y, padding=1):
    mask = y != 0
    batch_size, *spatial_dims = mask.shape
    valid_mask = mask.view(batch_size, -1).any(dim=1)

    # Expand valid_mask to match y shape
    expanded_valid_mask = valid_mask.view(batch_size, *([1] * (y.ndim - 1)))

    # If a batch has all zeros, do not apply masking (keep it as all True)
    mask = mask | ~expanded_valid_mask

    # Apply dilation (expanding nonzero regions)
    if padding > 0 and y.ndim == 3:
        mask = mask.float().unsqueeze(1)
        kernel = torch.ones(
            (1, 1, 2 * padding + 1, 2 * padding + 1),
            dtype=torch.float32,
            device=y.device,
        )
        mask = F.conv2d(mask, kernel, padding=padding).squeeze(1) > 0  # Binary mask

    return mask


def create_mask(y):
    # Create a boolean mask where y != 0
    mask = y != 0
    batch_size = y.shape[0]

    # Check if all elements are zero per batch
    valid_mask = mask.view(batch_size, -1).any(dim=1)

    # If a batch has all zeros, do not apply masking (keep it as all True)
    mask[~valid_mask] = True

    return mask


def get_loss_fn(kind="mse", *args, **kwargs):
    match kind:
        case "mse":
            return nn.MSELoss()
        case "dice":
            return BinaryDiceLoss()
        case "bce":
            return nn.BCEWithLogitsLoss()
        case "focal":
            return FocalLoss()
        case "mae" | "l1":
            return nn.L1Loss()
        case "huber":
            return nn.HuberLoss()
        case _:
            raise ValueError(f"Unrecognized loss function {kind}")


def get_scheduler(optimizer, kind="cosine", *args, **kwargs):
    match kind:
        case "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                kwargs.get("lr_T_max", 10),
                eta_min=kwargs.get("eta_min", 1e-4),
            )
        case "reduce_lr_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=kwargs.get("patience", 5),
                min_lr=kwargs.get("min_lr", 1e-4),
                factor=kwargs.get("factor", 0.5),
            )


def load_model(config: dict) -> ConvLSTM:
    input_channels = config.get("input_channels", 1)
    output_channels = config.get("output_channels", 1)
    kernel_size = tuple(config.get("kernel_size", (3, 3)))
    num_layers = config.get("num_layers", 1)
    model = ConvLSTM(
        input_dim=input_channels,
        hidden_dim=output_channels,
        kernel_size=kernel_size,
        num_layers=num_layers,
        batch_first=True,
        bias=True,
        return_all_layers=False,
    )
    return model


def create_config(args: dict) -> dict:
    args = {k: v for k, v in args.items() if v is not None}
    device = args.get("device", None)
    cpu_count = os.cpu_count() or 0
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    features_to_keep = args.get("features_to_keep")
    channel = 40 if features_to_keep is None else len(features_to_keep)  # type: ignore

    config = {
        "batch_size": args.get("batch_size", 1),
        "time_steps": args.get("time_steps", 5),
        "img_size": args.get("img_size", (128, 128)),
        "input_channels": channel,
        "output_channels": channel,
        "epochs": args.get("epochs", 100),
        "learning_rate": args.get("learning_rate", 0.01),
        "lr_T_max": args.get("lr_T_max", 30),
        "train_years": args.get("train_years", [2018, 2019]),
        "val_years": args.get("val_years", [2019]),
        "data_dir": args.get("data_dir", "./dataset_hdf5"),
        "features_to_keep": args.get("features_to_keep", None),
        "kernel_size": tuple(args.get("kernel_size", (3, 3))),
        "num_layers": args.get("num_layers", 2),
        "loss_fn": args.get("loss_fn", "mse"),
        "scheduler": args.get("scheduler", "cosine"),
        "num_workers": args.get("num_workers", cpu_count - 1),
        "device": device,
    }

    return config
