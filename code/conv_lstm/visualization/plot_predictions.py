import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from conv_lstm.utils import load_json_config, create_config, load_model
from conv_lstm.dataloader.FireSpreadDataset import FireSpreadDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on fire spread dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to dataset directory.",
        default="/home/petark/PycharmProjects/quantum-realtime-algorithm-for-wildfire-containment/conv_lstm",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../runs/run_13/1.pth",
        help="Path to trained model.",
    )
    parser.add_argument("--config", type=str, default="../config.json", help="Path to config file.")
    parser.add_argument(
        "--save_to",
        type=str,
        default="predictions",
        help="Directory to save predictions.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--time_steps", type=int, default=5, help="Number of leading observations.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=9e-5,
        help="Threshold for distinguishing active fire from background," " varies between checkpoints",
    )
    parser.add_argument("--val_year", type=int, default=2020, help="Year for validation data.")
    parser.add_argument("--device", default=None, help="Device for inference.")
    return parser.parse_args()


def load_dataset(data_dir, val_year, time_steps, img_size, features_to_keep):
    return FireSpreadDataset(
        data_dir=data_dir,
        included_fire_years=[val_year],
        n_leading_observations=time_steps,
        crop_side_length=img_size,
        load_from_hdf5=True,
        is_train=False,
        remove_duplicate_features=False,
        stats_years=[2018, 2019],
        features_to_keep=features_to_keep,
    )


def run_inference(model, dataloader, device, save_to, threshold=9e-5):
    os.makedirs(save_to, exist_ok=True)
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y_pred, _ = model(x)

            y_pred[y_pred < threshold] = 0
            y_pred[y_pred != 0] = 1
            y_pred_np = y_pred.cpu().numpy() * 255
            y_pred_np = y_pred_np.astype(np.uint8).transpose(1, 2, 0)
            y_np = y.cpu().numpy().astype(np.uint8).transpose(1, 2, 0) * 255
            fig, axs = plt.subplots(1, 7, figsize=(21, 5))
            for j in range(5):
                af = x[0, j, -1, :, :].cpu().numpy().astype(np.uint8)
                axs[j].imshow(af * 255, cmap="gray")
                axs[j].set_title(f"Input {j + 1}")

            axs[5].imshow(y_np, cmap="gray")
            axs[5].set_title("Ground Truth")
            axs[6].imshow(y_pred_np, cmap="gray")
            axs[6].set_title("Prediction")

            save_path = os.path.join(save_to, f"inputs_and_pred_{i}.png")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved: {save_path}")


def main():
    args = parse_args()
    json_config = load_json_config(args.config)
    config = create_config(json_config)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    dataset = load_dataset(
        config["data_dir"],
        config["val_years"][0],
        config["time_steps"],
        config["img_size"],
        config["features_to_keep"],
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    run_inference(model, dataloader, device, args.save_to, args.threshold)


if __name__ == "__main__":
    main()
