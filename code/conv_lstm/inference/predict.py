import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from conv_lstm.data.tiff import TiffDataset
from conv_lstm.utils import create_config, load_model, load_json_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fire_images_dir",
        type=str,
        default="./fires",
        help="path to fire .tif images",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="path to model checkpoint",
        default="../runs/run_13/1.pth",
    )
    parser.add_argument("--config", type=str, default="../config.json")
    parser.add_argument(
        "--save_to",
        type=str,
        default="output",
        help="path to save the model predictions," " if None, doesn't save",
    )
    parser.add_argument("--time_steps", type=int, default=5)
    parser.add_argument(
        "--threshold",
        type=float,
        default=9e-5,
        help="Threshold for distinguishing active fire from background," " varies between checkpoints",
    )
    return parser.parse_args()


class Predictor:
    def __init__(
        self,
        fire_images_dir: str,
        config: dict,
        model_path: str,
        threshold: float = 0.01,
        postprocess: bool = True,
        save_to: str | None = None,
        device=None,
    ):
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = load_model(config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.dataset = TiffDataset(
            fire_images_dir,
            time_steps=config["time_steps"],
            features_to_keep=config["features_to_keep"],
        )
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.threshold = threshold
        self.postprocess = postprocess
        self.save_to = save_to
        if self.save_to is not None:
            os.makedirs(self.save_to, exist_ok=True)

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                batch = batch.to(self.device)
                y_pred, _ = self.model(batch)
                yield y_pred

    def postprocess_prediction(self, y_pred):
        y_pred[y_pred < self.threshold] = 0
        y_pred[y_pred != 0] = 1
        return y_pred

    def __call__(self):
        for i, y_pred in enumerate(self.predict()):
            if self.postprocess:
                y_pred = self.postprocess_prediction(y_pred)
            y_pred = y_pred.cpu().numpy() * 255
            y_pred = y_pred.astype(np.uint8).transpose(1, 2, 0)
            if self.save_to:
                plt.imshow(y_pred, cmap="gray")
                plt.savefig(os.path.join(f"{self.save_to}", f"prediction_{i}.png"))
            yield y_pred


if __name__ == "__main__":
    args = parse_args()
    json_config = load_json_config(args.config)
    CONFIG = create_config(json_config)
    predictor = Predictor(
        fire_images_dir=args.fire_images_dir,
        config=CONFIG,
        model_path=args.model_path,
        threshold=args.threshold,
        save_to=args.save_to,
    )
    for i, prediction in enumerate(predictor()):
        print("Processed prediction #{}".format(i))
