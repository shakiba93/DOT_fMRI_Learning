import torch
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
from torch.utils.data import Sampler

## keep
class fNIRSPreloadDataset(Dataset):
    def __init__(self, data_csv_path, mode="train", chromo="HbO"):
        self.data_csv = pd.read_csv(data_csv_path)
        self.mode = mode
        self.chromo = chromo
        self.all_modalities = []

        # === Pre-load all trials into RAM ===
        self.all_trials = []
        self.all_labels = []

        print(f"Preloading {len(self.data_csv)} trials into memory...")
        print(self.chromo)
        for i, row in self.data_csv.iterrows():
            if chromo == "both":
                record = xr.open_dataarray(row["snirf_file"])
                trial_tensor = torch.tensor(record.values, dtype=torch.float32)
            else:
                try:
                    record = xr.open_dataarray(row["snirf_file"]).sel(chromo=chromo)
                    current_len = record.shape[1]
                    target_len = 87

                    # only pad if shorter than target
                    if current_len < target_len:
                        print("Padding trial from length", current_len, "to", target_len)
                        pad_width = [(0, 0), (0, target_len - current_len)]
                        record = xr.DataArray(
                            np.pad(record.values, pad_width, mode='constant', constant_values=0),
                            dims=record.dims,
                            coords={
                                record.dims[0]: record.coords[record.dims[0]].values,
                                record.dims[1]: np.arange(target_len)
                            }
                        )
                    trial_tensor = torch.tensor(record.values, dtype=torch.float32).unsqueeze(1)

                except Exception as e:
                    print(f"Error loading {row['snirf_file']}: {e}")
                    continue
            label_tensor = torch.tensor(int(row["trial_type"]), dtype=torch.long)

            self.all_trials.append(trial_tensor)
            self.all_labels.append(label_tensor)
            mod = 0 if row["modality"] == "fnirs" else 1
            self.all_modalities.append(torch.tensor(mod, dtype=torch.long))

        print(f"Loaded {len(self.all_trials)} trials into memory.")

    def __len__(self):
        return len(self.all_trials)

    def __getitem__(self, idx):
        # return torch.randn_like(self.all_trials[idx]), self.all_labels[idx], self.all_modalities[idx]
        return self.all_trials[idx], self.all_labels[idx], self.all_modalities[idx]


### keep
class ProportionalModalityBatchSampler(Sampler):
    def __init__(self, modalities, batch_size, fnirs_ratio=0.5, drop_last=True):
        """
        fnirs_ratio: fraction of fnirs in each batch (0–1)
        """
        assert 0 < fnirs_ratio < 1
        self.modalities = np.array(modalities)
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.fnirs_idx = np.where(self.modalities == 0)[0]
        self.fmri_idx  = np.where(self.modalities == 1)[0]

        self.n_fnirs = int(batch_size * fnirs_ratio)
        self.n_fmri  = batch_size - self.n_fnirs

        self.num_batches = min(
            len(self.fnirs_idx) // self.n_fnirs,
            len(self.fmri_idx)  // self.n_fmri
        )

    def __iter__(self):
        fnirs_perm = np.random.permutation(self.fnirs_idx)
        fmri_perm  = np.random.permutation(self.fmri_idx)

        for i in range(self.num_batches):
            batch = np.concatenate([
                fnirs_perm[i*self.n_fnirs:(i+1)*self.n_fnirs],
                fmri_perm[i*self.n_fmri:(i+1)*self.n_fmri]
            ])
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.num_batches
