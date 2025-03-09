import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from conv_lstm.model import ConvLSTM
from conv_lstm.dataloader.FireSpreadDataset import FireSpreadDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on single fire in loop.")
    parser.add_argument("--data_dir", type=str, help="Path to dataset directory.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../../runs/run_2/best.pth",
        help="Path to trained model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Directory to save predictions.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--time_steps", type=int, default=5, help="Number of leading observations.")
    parser.add_argument(
        "--img_size",
        type=tuple[int],
        default=(128, 128),
        help="Image crop side length.",
    )
    parser.add_argument("--val_year", type=int, default=2020, help="Year for validation data.")
    parser.add_argument("--index_to_plot", type=int, default=77, help="Which fire to plot.")
    parser.add_argument("--n_iters", type=int, default=50, help="Number of iterations to run.")
    return parser.parse_args()


def load_model(model_path, device, input_dim=40, hidden_dim=40, kernel_size=(3, 3), num_layers=1):
    model = ConvLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        num_layers=num_layers,
        batch_first=True,
        bias=True,
        return_all_layers=False,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


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


def plot_losses():
    if os.path.exists("../losses.csv"):
        losses = pd.read_csv("../losses.csv")
        losses.plot()
        plt.savefig("losses.png")
        plt.close()


def run_inference(model, dataloader, device, output_dir, index_to_plot, n_iters):
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            if i < index_to_plot:
                continue
            elif i > index_to_plot:
                break
            x = x.to(device)
            y = y.cpu()
            for j in range(n_iters):
                y_pred, _ = model(x)
                y_pred[y_pred < 1e-2] = 0
                y_pred[y_pred != 0] = 1
                y_pred_np = y_pred.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
                y_np = y.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)

                fig, axs = plt.subplots(1, 7, figsize=(21, 5))
                for k in range(5):
                    af = x[0, k, -1, :, :].cpu().numpy().astype(np.uint8)
                    axs[k].imshow(af * 255, cmap="gray")
                    axs[k].set_title(f"Input {k + 1}")

                axs[5].imshow(y_np, cmap="gray")
                axs[5].set_title("Ground Truth")
                axs[6].imshow(y_pred_np, cmap="gray")
                axs[6].set_title("Prediction")

                save_path = os.path.join(output_dir, f"inputs_and_pred_{i}_{j}.png")
                plt.savefig(save_path)
                plt.close(fig)
                print(f"Saved: {save_path}")
                x[0, :4, -1, :, :] = x[0, 1:5, -1, :, :].clone()
                x[0, 4, -1, :, :] = y_pred
                y = y_pred


def main():
    args = parse_args()
    args.features_to_keep = [-1]
    input_dim = 40 if args.features_to_keep is None else len(args.features_to_keep)
    hidden_dim = 40 if args.features_to_keep is None else len(args.features_to_keep)
    args.data_dir = "/home/petark/PycharmProjects/WildfireSpreadTS/dataset_hdf5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        args.model_path,
        device,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
    )
    dataset = load_dataset(
        args.data_dir,
        args.val_year,
        args.time_steps,
        args.img_size,
        args.features_to_keep,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    plot_losses()
    run_inference(model, dataloader, device, args.output_dir, args.index_to_plot, args.n_iters)


if __name__ == "__main__":
    main()
