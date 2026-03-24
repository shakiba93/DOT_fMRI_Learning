import pandas as pd
import numpy as np
import os
import glob
import random
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


# ======================================================
# HELPERS (GENERIC)
# ======================================================

def extract_label(filepath, label_map):
    fname = os.path.basename(filepath)

    if fname.endswith("_test.nc"):
        key = fname.split("_")[-3]
    else:
        key = fname.split("_")[-2]

    if key not in label_map:
        raise ValueError(f"Unknown label {key} in file {fname}")

    return label_map[key]


def get_modality(filepath):
    return "fmri" if "tfMRI" in filepath else "fnirs"


def collect_files(subjects, path, pattern="*.nc"):
    files = []
    for sub in subjects:
        files += glob.glob(os.path.join(path, sub, "**", pattern), recursive=True)
    return files


def build_dataframe(files, label_map, filter_labels=None):
    data = []

    for f in files:
        try:
            y = extract_label(f, label_map)
        except:
            continue

        if filter_labels is not None and y in filter_labels:
            continue

        data.append((f, y))

    df = pd.DataFrame(data, columns=["snirf_file", "trial_type"])
    df["modality"] = df["snirf_file"].apply(get_modality)

    return df


def load_participants(bids_path, preprocessed_path, pattern="sub-*"):
    if bids_path is not None:
        return pd.read_csv(os.path.join(bids_path, "participants.tsv"), sep="\t")
    else:
        subs = glob.glob(os.path.join(preprocessed_path, pattern))
        subs = [os.path.basename(s) for s in subs]
        return pd.DataFrame({"participant_id": subs})


# ======================================================
# MOTOR (fmri + fnirs)
# ======================================================
def split_motor(
    bids_path,
    preprocessed_path,
    test_subjects_list,
    exclude_subjects=None,
    train_dataset="yuanyuan",
    fmri_subjects=24
):

    LABELS = {"Left": 0, "Right": 1, "left": 0, "right": 1}

    yuanyuan = ['sub-177', 'sub-182', 'sub-185', 'sub-633', 'sub-176', 'sub-580', 'sub-583', 'sub-586', 'sub-618', 'sub-640', 'sub-568', 'sub-621']
    laura = ['sub-179', 'sub-183', 'sub-581', 'sub-181', 'sub-587', 'sub-577', 'sub-638', 'sub-619', 'sub-613', 'sub-592', 'sub-170', 'sub-173']

    df = load_participants(bids_path, preprocessed_path)

    if exclude_subjects:
        df = df[~df["participant_id"].isin(exclude_subjects)]

    all_subs = df["participant_id"].values

    # -------------------------
    # dataset selection
    # -------------------------
    if train_dataset == "yuanyuan":
        data_list = yuanyuan
        name = "baseline"

    elif train_dataset == "laura":
        data_list = laura + yuanyuan
        name = "fnirs"

    elif train_dataset == "fmri":
        fmri = [s for s in all_subs if s not in laura + yuanyuan]
        fmri_data = random.sample(fmri, int(fmri_subjects))
        data_list = laura + yuanyuan + fmri_data
        name = f"fmri_{fmri_subjects}"

    else:
        raise ValueError("Unknown train_dataset")

    train_subs = [s for s in all_subs if s not in test_subjects_list and s in data_list]

    # -------------------------
    # build df
    # -------------------------
    train_files = collect_files(train_subs, preprocessed_path, "*_test.nc")
    train_df = build_dataframe(train_files, LABELS)

    # fnirs split
    val_subs = [s for s in train_subs if s in yuanyuan + laura]
    fnirs_df = train_df[train_df["snirf_file"].str.contains("|".join(val_subs))]

    train_fnirs, val_fnirs = train_test_split(
        fnirs_df,
        test_size=0.2,
        stratify=fnirs_df["trial_type"],
        random_state=42
    )

    fmri_df = train_df.drop(fnirs_df.index)

    train_df = pd.concat([train_fnirs, fmri_df])
    val_df = val_fnirs

    test_files = collect_files(test_subjects_list, preprocessed_path, "*_test.nc")
    test_df = build_dataframe(test_files, LABELS)

    return train_df, val_df, test_df


# ======================================================
# WUSTL (binary)
# ======================================================
def split_wustl(
    bids_path,
    preprocessed_path,
    test_subjects_list,
    val_subjects_list,
    exclude_subjects=None
):

    LABELS = {"OV": 0, "CV": 1, "rest": 3, "RW": 3, "MEMa1": 3}

    df = load_participants(bids_path, preprocessed_path)

    if exclude_subjects:
        df = df[~df["participant_id"].isin(exclude_subjects)]

    all_subs = df["participant_id"].values

    train_subs = [
        s for s in all_subs
        if s not in test_subjects_list and s not in val_subjects_list
    ]

    def collect(subs):
        files = collect_files(subs, preprocessed_path, "*_test.nc")
        df = build_dataframe(files, LABELS, filter_labels=[3])
        df["modality"] = "fnirs"
        return df

    train_df = collect(train_subs)
    val_df = collect(val_subjects_list)
    test_df = collect(test_subjects_list)

    return train_df, val_df, test_df


# ======================================================
# WUSTL MULTI
# ======================================================
def split_wustl_multi(
    bids_path,
    preprocessed_path,
    test_subjects_list,
    val_trial_percentage=0.2
):

    LABELS = {
        "ACL": 0, "ACR": 1, "MOTR": 2, "MOTL": 3,
        "HW": 4, "GV": 7, "MOV": 7, "rest": 7
    }

    df = load_participants(bids_path, preprocessed_path, pattern="subj-*")

    all_subs = df["participant_id"].values
    train_subs = [s for s in all_subs if s not in test_subjects_list]

    def collect(subs, mode="train"):
        pattern = "*.nc" if mode == "train" else "*_test.nc"
        files = collect_files(subs, preprocessed_path, pattern)
        df = build_dataframe(files, LABELS, filter_labels=[7])
        df["modality"] = "fnirs"
        return df

    test_df = collect(test_subjects_list, mode="test")
    train_pool = collect(train_subs)

    # group split (prevent leakage)
    def get_group_id(path):
        fname = os.path.basename(path)
        return fname.replace("_test.nc", "").replace(".nc", "")

    train_pool["group_id"] = train_pool["snirf_file"].apply(get_group_id)

    group_df = train_pool.groupby("group_id").first().reset_index()

    y = group_df["trial_type"].values
    idx = np.arange(len(group_df))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_trial_percentage, random_state=42)
    train_idx, val_idx = next(sss.split(idx, y))

    train_groups = set(group_df.iloc[train_idx]["group_id"])
    val_groups = set(group_df.iloc[val_idx]["group_id"])

    train_df = train_pool[train_pool["group_id"].isin(train_groups)]
    val_df = train_pool[train_pool["group_id"].isin(val_groups)]

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df