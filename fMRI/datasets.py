import torch
from torch.utils.data import Dataset
import h5py
import os


class HCPTrialDataset(Dataset):
    def __init__(self, trial_dir):
        self.files = sorted(
            os.path.join(trial_dir, f)
            for f in os.listdir(trial_dir)
            if f.endswith(".h5")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with h5py.File(self.files[idx], "r") as f:
            x = torch.from_numpy(f["x"][:]).float()
            y = torch.tensor(f["label"][()], dtype=torch.long)
        return x, y

