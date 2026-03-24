import os, glob, pickle, argparse
import numpy as np
import pandas as pd
import xarray as xr
import cedalion

import cedalion.sigproc.motion_correct as motion_correct
import cedalion.sigproc.quality as quality
import cedalion.sigproc.physio as physio
import cedalion.nirs as nirs

from cedalion.io.forward_model import load_Adot
import cedalion.dot as dot
from cedalion import units

from scipy.signal import resample_poly
from fractions import Fraction
from cedalion.dot.utils import chunked_eff_xr_matmult


# ============================================================
# --------------------- RESAMPLING ----------------------------
# ============================================================
def resample_signal(signal, tr_old, tr_new):
    fs_old = 1.0 / tr_old
    fs_new = 1.0 / tr_new

    ratio = Fraction(fs_new / fs_old).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator

    return resample_poly(signal, up, down)


def change_freq(da, tr_new):
    t_old = da["time"].values
    tr_old = float(np.mean(np.diff(t_old)))

    n_new = int(np.round((t_old[-1] - t_old[0]) / tr_new)) + 1
    t_new = t_old[0] + np.arange(n_new) * tr_new

    out = np.empty((n_new, da.sizes["parcel"], da.sizes["chromo"]), dtype=np.float32)

    for p in range(da.sizes["parcel"]):
        for c in range(da.sizes["chromo"]):
            sig = da[:, p, c].values
            rs = resample_signal(sig, tr_old, tr_new)

            if rs.shape[0] >= n_new:
                out[:, p, c] = rs[:n_new]
            else:
                out[:rs.shape[0], p, c] = rs
                out[rs.shape[0]:, p, c] = 0

    da_new = xr.DataArray(
        out,
        dims=("time", "parcel", "chromo"),
        coords={"time": t_new, "parcel": da.parcel, "chromo": da.chromo},
    )

    da_new = da_new.assign_coords(samples=("time", np.arange(n_new)))
    da_new.time.attrs["units"] = "s"

    return da_new


# ============================================================
# --------------------- HELPERS -------------------------------
# ============================================================
def get_bad_ch_mask(int_data):
    _, amp_mask_sat = quality.mean_amp(int_data, [0., 0.84])
    _, amp_mask_low = quality.mean_amp(int_data, [1e-3, 1])
    _, snr_mask = quality.snr(int_data, 10)

    amp_mask = amp_mask_sat & amp_mask_low
    _, bad = quality.prune_ch(int_data, [amp_mask, snr_mask], "all")

    return bad


# ============================================================
# --------------------- MAIN --------------------------------
# ============================================================
def process_file(file, args, recon_model, recon_type, sensitive_parcels):

    records = cedalion.io.read_snirf(file)
    rec = records[0]

    # -------- EVENTS --------
    if args.dataset == "laura":
        try:
            rec.stim = pd.read_csv(file.replace("nirs.snirf", "events.tsv"), sep="\t")
            rec.stim = rec.stim.sort_values("onset")
        except:
            return
    else:
        rec.stim = rec.stim.sort_values("onset")

    if rec.stim.shape[0] == 0:
        return

    # -------- AMP → OD --------
    if args.dataset != "wustl":
        rec["rep_amp"] = quality.repair_amp(rec["amp"], median_len=3, method="linear")
        rec["od_amp"], baseline = nirs.cw.int2od(rec["rep_amp"], return_baseline=True)
    else:
        rec["od_amp"], baseline = nirs.cw.int2od(rec["amp"], return_baseline=True)

    # -------- MOTION --------
    rec["od_tddr"] = motion_correct.tddr(rec["od_amp"])
    rec["od_tddr_wavel"] = motion_correct.wavelet(rec["od_tddr"])

    # -------- FILTER --------
    rec["od_hpfilt"] = rec["od_tddr_wavel"].cd.freq_filter(fmin=0.008, fmax=0, butter_order=4)

    # -------- CLEAN AMP --------
    rec["amp_clean"] = nirs.cw.od2int(rec["od_hpfilt"], baseline)

    # -------- CHANNEL PRUNING --------
    ds_ch, chd_mask = quality.sd_dist(rec["amp_clean"], rec.geo3d, [1, 4.5]*units.cm)
    rec["amp_clean"], _ = quality.prune_ch(rec["amp_clean"], [chd_mask], "all")

    bad_ch = get_bad_ch_mask(rec["amp_clean"])

    # -------- CONVERSION --------
    dpf = xr.DataArray([6, 6], dims="wavelength", coords={"wavelength": rec["amp"].wavelength})
    rec["conc"] = nirs.cw.od2conc(rec["od_hpfilt"], rec.geo3d, dpf, spectrum="prahl")

    chromo_var = quality.measurement_variance(rec["conc"], bad_ch, 1e6, False)
    rec["conc_pcr"], _ = physio.global_component_subtract(rec["conc"], ts_weights=1/chromo_var)

    rec["od_pcr1"] = nirs.cw.conc2od(rec["conc_pcr"], rec.geo3d, dpf, spectrum="prahl")

    c_meas = quality.measurement_variance(rec["od_hpfilt"], bad_ch, 1e6, False)

    # ============================================================
    # -------- RECONSTRUCTION (FIXED) -----------------------------
    # ============================================================
    if recon_type == "dot":
        delta = recon_model.reconstruct(rec["od_pcr1"], c_meas)

    elif recon_type == "matrix":
        delta = chunked_eff_xr_matmult(
            recon_model,
            rec["od_pcr1"],
            dim_src="channel",
            dim_dst="parcel"
        )

    delta.time.attrs["units"] = units.s

    # -------- KEEP TRUE PARCELS --------
    delta = delta.where(delta.is_brain == True)
    delta = delta.pint.quantify().pint.to("uM").pint.dequantify()

    # -------- PARCEL --------
    hbo = delta.sel(chromo="HbO").groupby("parcel").mean()
    hbr = delta.sel(chromo="HbR").groupby("parcel").mean()

    signal = xr.concat([hbo, hbr], dim="chromo")

    signal = signal.sel(parcel=signal.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_LH')
    signal = signal.sel(parcel=signal.parcel != 'Background+FreeSurfer_Defined_Medial_Wall_RH')
    signal = signal.sel(parcel=sensitive_parcels)

    delta, _ = physio.global_component_subtract(signal)

    # -------- TIME --------
    t_start = rec.stim.onset.min() - 3
    t_end   = rec.stim.onset.max() + 12

    t_start = max(0, t_start)
    t_end   = min(float(delta.time.max()), t_end)

    delta = delta.transpose("time", "parcel", "chromo")
    delta = change_freq(delta, 0.115)

    baseline = delta.groupby("chromo").mean(dim=("time", "parcel"))
    delta = delta - baseline

    delta = delta.cd.freq_filter(fmin=0.01, fmax=0.5, butter_order=4)
    delta = delta.sel(time=slice(t_start, t_end))

    hbr = delta.sel(chromo="HbR")
    E = np.sqrt((hbr**2).mean(dim=("time", "parcel")))
    delta = delta / (E + 1e-8)

    # -------- SEGMENT --------
    i = 0

    for _, row in rec.stim.iterrows():
        label = row["trial_type"]

        s = 0.0  # centered only

        start = row["onset"] + s
        end = start + 15

        baseline = delta.sel(
            time=slice(row["onset"] - 2.5, row["onset"])
        ).mean("time")

        x = delta.sel(time=slice(start, end)) - baseline

        x = x.isel(time=slice(0, 87))
        x = x.transpose("parcel", "chromo", "time")

        out = file.replace(args.input, args.output)
        os.makedirs(os.path.dirname(out), exist_ok=True)

        suffix = "_test"

        x.to_netcdf(out.replace(".snirf", f"_{label}_{i}{suffix}.nc"))

        i += 1

# ============================================================
# --------------------- CLI ---------------------------------
# ============================================================
def load_reconstruction_model(path, typ):
    if typ == "h5":
        Adot = load_Adot(path)
        return dot.ImageRecon(Adot, recon_mode="mua2conc", brain_only=True), "dot"

    elif typ == "pkl":
        with open(path, "rb") as f:
            B = pickle.load(f)
        return B, "matrix"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True, choices=["laura", "yuanyuan", "wustl"])
    parser.add_argument("--sensitivity", required=True)
    parser.add_argument("--sensitivity_type", required=True, choices=["h5", "pkl"])
    parser.add_argument("--sensitive_parcels", required=True)

    args = parser.parse_args()

    with open(args.sensitive_parcels, "rb") as f:
        sensitive_parcels = pickle.load(f)

    recon_model, recon_type = load_reconstruction_model(
        args.sensitivity,
        args.sensitivity_type
    )

    files = glob.glob(args.input + "/**/*.snirf", recursive=True)

    for f in files:
        process_file(f, args, recon_model, recon_type, sensitive_parcels)