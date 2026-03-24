import os
import argparse
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

import nibabel as nib
import xarray as xr
from scipy.interpolate import interp1d
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
import pickle
import cedalion


# ============================================================
# ---------------------- CONFIG -------------------------------
# ============================================================

TR = 0.72
TR_NEW = 0.115

WINDOW_LEN = 12.0
BASELINE_LEN = 2.5
TARGET_T = 87

DTYPE = np.float16


# -------- 15-CLASS SETUP --------
EVENT_TYPES_15 = [
    "rh", "lh", "rf", "lf", "t",
    "story", "math", "loss", "win",
    "match", "relation", "rnd", "mental",
    "2bk", "0bk"
]

EVENT_TO_IDX_15 = {e: i for i, e in enumerate(EVENT_TYPES_15)}


def collapse_label(trial_type: str) -> str:
    if trial_type.startswith("2bk_"):
        return "2bk"
    elif trial_type.startswith("0bk_"):
        return "0bk"
    else:
        return trial_type


# ============================================================
# ------------------- STAGE 1: FMRI → HEMO --------------------
# ============================================================

def increase_freq(t_old, signal, tr_new):
    t_new = np.arange(t_old[0], t_old[-1], tr_new)
    f = interp1d(t_old, signal, kind="cubic", axis=-1, fill_value="extrapolate")
    return f(t_new), t_new


def process_fmri_to_hemo(input_nii, output_h5, sensitive_parcels):

    if os.path.exists(output_h5):
        print(f"[SKIP] Already exists: {output_h5}")
        return

    # ---------------- Atlas ----------------
    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=600, yeo_networks=17, resolution_mm=2
    )

    atlas_img = schaefer.maps
    labels = [l.decode("utf-8") for l in schaefer.labels]

    new_labels = []
    for l in labels:
        l = l.replace("17Networks_", "")
        hem = l.split("_")[0]
        if hem == "LH":
            new_labels.append(l.replace("LH_", "") + "_LH")
        elif hem == "RH":
            new_labels.append(l.replace("RH_", "") + "_RH")

    parcel_indices = [new_labels.index(p) for p in sensitive_parcels]

    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=False,
        detrend=False,
        low_pass=None,
        high_pass=None,
        t_r=TR
    )

    # ---------------- Load fMRI ----------------
    img = nib.load(input_nii)
    ts = masker.fit_transform(img)          # time × parcels
    ts = ts[:, parcel_indices].T            # parcels × time

    bold_ds = xr.DataArray(
        ts,
        dims=["parcel", "time"],
        coords={
            "parcel": sensitive_parcels,
            "time": np.arange(ts.shape[1]) * TR,
        },
    )

    # ---------------- Safety trimming (RESTORED) ----------------
    t_start = 3.0
    t_end   = float(bold_ds.time.max()) - 5

    # ---------------- HbR ----------------
    hbr = 1.0 / (bold_ds + 1e-8)
    x = hbr.mean("time")
    hbr = (hbr - x) / x

    # ---------------- HbO ----------------
    shift_sec = 2.0
    shift_samples = int(shift_sec / TR)

    hbo = -3.0 * hbr
    hbo = hbo.roll(time=-shift_samples, roll_coords=False)
    hbo = hbo.where(hbo.time <= hbo.time.values[-shift_samples], 0)

    hemo = xr.concat([hbo, hbr], dim="chromo")
    hemo = hemo.assign_coords(chromo=["HbO", "HbR"])

    # ---------------- Upsampling ----------------
    hemo_vals, t_new = increase_freq(
        hemo.time.values,
        hemo.transpose("chromo", "parcel", "time").values,
        TR_NEW
    )

    hemo = xr.DataArray(
        hemo_vals,
        dims=["chromo", "parcel", "time"],
        coords={
            "chromo": ["HbO", "HbR"],
            "parcel": hemo.parcel.values,
            "time": t_new,
        },
    )

    # ---------------- Filtering ----------------
    hemo = hemo.rename({"parcel": "channel"})
    hemo = hemo.assign_coords(samples=("time", range(len(hemo["time"]))))
    hemo.time.attrs["units"] = "seconds"
    hemo = hemo.cd.freq_filter(fmin=0.01, fmax=0.5, butter_order=4)
    hemo = hemo.rename({"channel": "parcel"})

    # ---------------- Time trimming AFTER filtering (RESTORED) ----------------
    hemo = hemo.sel(time=slice(t_start, t_end))

    # ---------------- Remove medial wall ----------------
    hemo = hemo.sel(parcel=hemo.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_LH')
    hemo = hemo.sel(parcel=hemo.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_RH')

    # ---------------- Normalization ----------------
    hbr = hemo.sel(chromo="HbR")
    E = np.sqrt((hbr**2).mean(dim=("time", "parcel")))
    hemo = hemo / (E + 1e-8)

    # ---------------- Save ----------------
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)

    with h5py.File(output_h5, "w") as hf:
        hf.create_dataset("data", data=hemo.values.astype(np.float32))
        hf.create_dataset("time", data=hemo.coords["time"].values.astype(np.float32))
        hf.create_dataset("chromo", data=np.array(["HbO", "HbR"], dtype="S"))

    print(f"[OK] Saved: {output_h5}")


# ============================================================
# ------------------- STAGE 2: HEMO → TRIALS ------------------
# ============================================================

def extract_trial(f, chromo_indices, onset):
    time = f["time"][:]

    start = onset
    end = onset + WINDOW_LEN
    baseline_start = onset - BASELINE_LEN

    time_idxs = np.where((time >= start) & (time < end))[0]
    baseline_idxs = np.where((time >= baseline_start) & (time < onset))[0]

    if len(time_idxs) < TARGET_T or len(baseline_idxs) == 0:
        return None

    trial_tmp = f["data"][:, :, time_idxs[:TARGET_T]]
    baseline_tmp = f["data"][:, :, baseline_idxs]

    signal = trial_tmp[chromo_indices]
    baseline = baseline_tmp[chromo_indices].mean(axis=2, keepdims=True)

    x = signal - baseline
    x = np.nan_to_num(x)

    return x.astype(DTYPE)


def extract_trials(csv_path, out_dir, chromos=("HbO", "HbR")):

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"[INFO] Extracting {len(df)} trials")

    checked_shape = False

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        try:
            with h5py.File(row["record_file"], "r") as f:

                chromo_names = f["chromo"][:].astype(str).tolist()
                chromo_indices = [chromo_names.index(c) for c in chromos]

                x = extract_trial(f, chromo_indices, float(row["onset"]))
                if x is None:
                    continue

                # -------- SHAPE CHECK (RESTORED) --------
                if not checked_shape:
                    assert x.shape == (len(chromo_indices), x.shape[1], TARGET_T)
                    checked_shape = True

                # -------- LABEL (15-class) --------
                label_name = collapse_label(row["trial_type"])
                label = EVENT_TO_IDX_15[label_name]

                # -------- SAVE --------
                trial_name = f"{row['subject_id']}_{label_name}_{idx:07d}.h5"
                out_path = os.path.join(out_dir, trial_name)

                with h5py.File(out_path, "w") as out:
                    out.create_dataset("x", data=x, compression="lzf", shuffle=True)
                    out.create_dataset("label", data=label)
                    out.create_dataset("onset", data=row["onset"])

                    # -------- METADATA (RESTORED) --------
                    out.create_dataset("trial_type", data=str(label_name).encode())
                    out.create_dataset("subject_id", data=str(row["subject_id"]).encode())
                    out.create_dataset("record_file", data=str(row["record_file"]).encode())
                    out.create_dataset("chromo", data=np.array(chromos, dtype="S"))

        except Exception as e:
            print(f"[WARN] Skipped {idx}: {e}")

    print("[DONE] Trials saved")


# ============================================================
# ------------------------- CLI -------------------------------
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--stage", choices=["hemo", "trials"], required=True)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--csv")
    parser.add_argument("--sensitive_parcels")

    args = parser.parse_args()

    if args.stage == "hemo":
        with open(args.sensitive_parcels, "rb") as f:
            sensitive_parcels = pickle.load(f)

        process_fmri_to_hemo(
            input_nii=args.input,
            output_h5=args.output,
            sensitive_parcels=sensitive_parcels
        )

    elif args.stage == "trials":
        extract_trials(
            csv_path=args.csv,
            out_dir=args.output
        )