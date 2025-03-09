import argparse
import json
import os
import numpy as np
import torch
import pandas as pd
from torch import optim
from tqdm import tqdm

from conv_lstm.utils import (
    get_loss_fn,
    get_scheduler,
    load_json_config,
    load_model,
    create_config,
)

from torch.utils.data import DataLoader, Subset
from conv_lstm.dataloader.FireSpreadDataset import FireSpreadDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    return parser.parse_args()


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Trainer:
    def __init__(
        self,
        epochs,
        model,
        optimizer,
        criterion,
        scheduler,
        train_years,
        val_years,
        num_workers,
        data_dir,
        time_steps,
        crop_side_length,
        stats_years,
        features_to_keep,
        early_stopper: EarlyStopper | None = None,
        device: str | torch.device = "cuda:0",
    ):

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.early_stopper = early_stopper or EarlyStopper()
        self.train_years = train_years
        self.val_years = val_years
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.time_steps = time_steps
        self.crop_side_length = crop_side_length
        self.stats_years = stats_years
        self.features_to_keep = features_to_keep
        self.epochs = epochs

    # Training function
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        for x, y in tqdm(train_loader, total=len(train_loader), desc="Training"):
            x, y = x.float(), y.float()
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            y_pred, _ = self.model(x)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    # Validation function
    def validate_epoch(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validation"):
                x, y = x.float(), y.float()
                x, y = x.to(self.device), y.to(self.device)
                y_pred, _ = self.model(x)
                val_loss += self.criterion(y_pred, y).item()
        return val_loss / len(val_loader)

    def train_loop(self, CONFIG, train_loader, val_loader):
        # Training loop
        run_dir = self.create_new_run_folder(CONFIG)
        train_losses, val_losses = [], []

        best_loss = np.inf
        lr = self.scheduler.get_last_lr()[0]
        print("Starting learning rate:", lr)
        best_path = os.path.join(run_dir, "best.pth")
        last_path = os.path.join(run_dir, "last.pth")

        for epoch in range(CONFIG["epochs"]):
            epoch_path = os.path.join(run_dir, f"{epoch}.pth")
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            self.scheduler.step()

            new_lr = self.scheduler.get_last_lr()[0]
            if new_lr != lr:
                print(f"Adjusting learning rate from {lr} to {new_lr}")
                lr = new_lr

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                print(f"New best validation loss: {best_loss}")
                torch.save(self.model.state_dict(), best_path)
            print(
                f"Epoch {epoch + 1}/{CONFIG['epochs']} - \
                    Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            torch.save(self.model.state_dict(), best_path)
            if self.early_stopper.early_stop(val_loss):
                print("Early stopping")
                break
            torch.save(self.model.state_dict(), epoch_path)

        # Save the model and losses
        torch.save(self.model.state_dict(), last_path)
        pd.DataFrame(np.c_[train_losses, val_losses], columns=["train_loss", "val_loss"]).to_csv(
            "runs/run_13/losses.csv", index=False
        )

    # Used for saving the model
    def create_new_run_folder(self, CONFIG, base_dir="runs"):
        os.makedirs(base_dir, exist_ok=True)

        # Find the next available index
        existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run_") and d[4:].isdigit()]
        run_indices = [int(d[4:]) for d in existing_runs]
        next_index = max(run_indices, default=-1) + 1

        new_folder = os.path.join(base_dir, f"run_{next_index}")
        os.makedirs(new_folder)

        config_path = os.path.join(new_folder, "config.json")
        with open(config_path, "w") as f:
            json.dump(CONFIG, f, indent=4)

        return new_folder

    def get_dataloader(
        self,
        batch_size,
        is_train=True,
        subset_size=None,
        shuffle=True,
        num_workers=1,
        load_from_hdf5=True,
    ) -> DataLoader:
        years = self.train_years if is_train else self.val_years

        dataset = FireSpreadDataset(
            data_dir=self.data_dir,
            included_fire_years=years,
            n_leading_observations=self.time_steps,
            crop_side_length=self.crop_side_length,
            load_from_hdf5=load_from_hdf5,
            is_train=is_train,
            remove_duplicate_features=False,
            stats_years=self.train_years,
            features_to_keep=self.features_to_keep,
        )

        if subset_size:
            indices = list(range(subset_size))
            dataset = Subset(dataset, indices)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    args = parse_args()
    json_config = load_json_config(args.config)
    CONFIG = create_config(json_config)
    os.makedirs("runs", exist_ok=True)

    model = load_model(CONFIG)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = get_loss_fn(kind=CONFIG["loss_fn"])
    scheduler = get_scheduler(optimizer, kind=CONFIG["scheduler"])
    early_stopper = EarlyStopper(patience=10, min_delta=0)

    trainer = Trainer(
        early_stopper=early_stopper,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=CONFIG["device"],
        epochs=CONFIG["epochs"],
        train_years=CONFIG["train_years"],
        val_years=CONFIG["val_years"],
        num_workers=CONFIG["num_workers"],
        data_dir=CONFIG["data_dir"],
        time_steps=CONFIG["time_steps"],
        crop_side_length=CONFIG["img_size"][0],
        stats_years=CONFIG["train_years"],
        features_to_keep=CONFIG["features_to_keep"],
    )

    train_loader = trainer.get_dataloader(
        batch_size=CONFIG["batch_size"],
        is_train=True,
        subset_size=None,
    )

    val_loader = trainer.get_dataloader(is_train=False, subset_size=None, batch_size=1, shuffle=False)

    trainer.train_loop(CONFIG, train_loader, val_loader)
