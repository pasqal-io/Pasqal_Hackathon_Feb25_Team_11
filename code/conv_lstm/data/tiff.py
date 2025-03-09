import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TiffDataset(Dataset):
    def __init__(self, folder_path, features_to_keep=None, time_steps=5):
        self.folder_path = folder_path
        self.time_steps = time_steps
        self.files = sorted(os.listdir(folder_path))
        self.indices = self._generate_indices()
        self.features_to_keep = features_to_keep

    def _generate_indices(self):
        return [self.files[i : i + self.time_steps] for i in range(len(self.files) - self.time_steps + 1)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_files = self.indices[idx]
        sample = []
        for file in sample_files:
            file_path = os.path.join(self.folder_path, file)
            with rasterio.open(file_path) as src:
                img = src.read().astype(np.float32)  # Shape: (C, H, W)
                if self.features_to_keep is not None:
                    img = img[self.features_to_keep, ...]
                img[np.isnan(img)] = 0  # Replace NaNs with 0
                img[img != 0] = 1  # Binarization
                sample.append(img)
        sample = np.stack(sample)  # Shape: (window_size, C, H, W)
        return torch.tensor(sample, dtype=torch.float32)


if __name__ == "__main__":
    # Usage
    folder_path = (
        "/home/petark/PycharmProjects/quantum-realtime-algorithm-for-wildfire-containment/fires_kotec/fires"
    )
    time_steps = 5
    dataset = TiffDataset(folder_path, time_steps=time_steps)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(len(dataloader))
    # Fetch a batch
    for batch in dataloader:
        print(batch.shape)  # Expected shape: (B, T, C, H, W) where T = window_size
        break
