import os, sys, pickle, argparse
import numpy as np
import nibabel as nib
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
import cedalion


# -----------------------
# Utils
# -----------------------
def increase_freq(t_old, signal, tr_new):
    t_new = np.arange(t_old[0], t_old[-1], tr_new)
    f = interp1d(t_old, signal, kind="cubic", axis=-1, fill_value="extrapolate")
    return f(t_new)


# -----------------------
# GLOBAL ATLAS (ONLY ONCE)
# -----------------------
def load_atlas_and_masker(sensitive_parcels):

    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=600, yeo_networks=17, resolution_mm=2
    )

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
        labels_img=schaefer.maps,
        standardize=False,
        detrend=False,
        low_pass=None,
        high_pass=None,
        t_r=0.72
    )

    return masker, parcel_indices


# -----------------------
# MAIN PROCESS FUNCTION
# -----------------------
def process_file(f, dataset_path, processed_data, masker, parcel_indices):

    print("Processing:", f)

    tr = 0.72
    tr_new = 0.115

    out_file = f.replace(dataset_path, processed_data).replace(".nii.gz", ".nc")
    if os.path.exists(out_file):
        print("Already processed.")
        return

    # ---------------- Load fMRI ----------------
    img = nib.load(f)

    ts = masker.fit_transform(img)          # time × parcels
    ts = ts[:, parcel_indices].T            # parcels × time

    bold_ds = xr.DataArray(
        ts,
        dims=["parcel", "time"],
        coords={
            "parcel": np.array(parcel_indices),  # same indexing behavior
            "time": np.arange(ts.shape[1]) * tr,
        },
    )

    # ---------------- Load events ----------------
    base = os.path.dirname(f)

    lh = pd.read_csv(os.path.join(base, "EVs/lh.txt"), sep="\t", header=None)
    lh.columns = ["onset", "duration", "value"]
    lh["trial_type"] = "left"

    rh = pd.read_csv(os.path.join(base, "EVs/rh.txt"), sep="\t", header=None)
    rh.columns = ["onset", "duration", "value"]
    rh["trial_type"] = "right"

    events_df = pd.concat([lh, rh]).sort_values("onset")
    events_df = events_df[events_df.onset > 0]

    # ---------------- Crop window ----------------
    t_start = events_df.onset.min() - 3.0
    t_end   = events_df.onset.max() + 20.0

    t_start = max(0, t_start)
    t_end   = min(float(bold_ds.time.max()), t_end)

    # ============================================================
    # Continuous HbR proxy
    # ============================================================
    hbr_cont = 1.0 / (bold_ds + 1e-8)
    x = hbr_cont.mean("time")
    hbr_cont = (hbr_cont - x) / x

    # ============================================================
    # Continuous HbO proxy
    # ============================================================
    time_shift_seconds = 2.0
    scale_factor = 3.0
    fs = 1 / tr
    shift_samples = int(time_shift_seconds * fs)

    hbo_cont = -scale_factor * hbr_cont
    hbo_cont = hbo_cont.roll(time=-shift_samples, roll_coords=False)
    hbo_cont = hbo_cont.where(
        hbo_cont.time <= hbo_cont.time.values[-shift_samples], 0
    )

    hemo_cont = xr.concat([hbo_cont, hbr_cont], dim="chromo")
    hemo_cont = hemo_cont.assign_coords(chromo=["HbO", "HbR"])

    # ---------------- Upsample ----------------
    t_old = hemo_cont.time.values
    t_new = np.arange(t_old[0], t_old[-1], tr_new)

    hemo_cont_tlast = hemo_cont.transpose("chromo", "parcel", "time")
    hemo_up = increase_freq(t_old, hemo_cont_tlast.values, tr_new)

    hemo_cont = xr.DataArray(
        hemo_up,
        dims=["chromo", "parcel", "time"],
        coords={
            "chromo": hemo_cont.chromo.values,
            "parcel": hemo_cont.parcel.values,
            "time": t_new,
        },
    )

    # ---------------- Filtering ----------------
    hemo_cont = hemo_cont.rename({"parcel": "channel"})
    hemo_cont = hemo_cont.assign_coords(samples=("time", range(len(hemo_cont["time"]))))
    hemo_cont.time.attrs["units"] = "seconds"
    hemo_cont = hemo_cont.cd.freq_filter(fmin=0.01, fmax=0.5, butter_order=4)
    hemo_cont = hemo_cont.rename({"channel": "parcel"})

    hemo_cont = hemo_cont.sel(time=slice(t_start, t_end))

    # ---------------- Remove medial wall ----------------
    hemo_cont = hemo_cont.sel(parcel=hemo_cont.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_LH')
    hemo_cont = hemo_cont.sel(parcel=hemo_cont.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_RH')

    # ---------------- Normalization ----------------
    hbr = hemo_cont.sel(chromo='HbR')
    E = np.sqrt((hbr**2).mean(dim=("time", "parcel")))
    hemo_cont = hemo_cont / (E + 1e-8)

    # ============================================================
    # Segmentation (NO JITTER)
    # ============================================================
    i = 0

    for _, row in events_df.iterrows():

        label = row.trial_type

        start = row.onset
        end = start + 15

        x = hemo_cont.sel(time=slice(start, end))

        baseline = hemo_cont.sel(
            time=slice(row["onset"] - 2.5, row["onset"])
        ).mean("time")

        x = x - baseline

        x = x.isel(time=slice(0, 87))
        x = x.transpose("parcel", "chromo", "time")

        out_dir = os.path.join(processed_data, f.split("/")[3], "nirs")
        os.makedirs(out_dir, exist_ok=True)

        out = os.path.join(out_dir, os.path.basename(f))
        x.to_netcdf(out.replace(".nii.gz", f"_{label}_{i}.nc"))

        i += 1

    print("Finished:", os.path.basename(f))


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--sensitive_parcels", required=True)

    args = parser.parse_args()

    with open(args.sensitive_parcels, "rb") as f:
        sensitive_parcels = pickle.load(f)

    # ✅ load ONCE
    masker, parcel_indices = load_atlas_and_masker(sensitive_parcels)

    process_file(
        f=args.input,
        dataset_path=args.dataset_path,
        processed_data=args.output_path,
        masker=masker,
        parcel_indices=parcel_indices
    )