import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score

from datasets_v02 import fNIRSPreloadDataset, ProportionalModalityBatchSampler
from model import CNN2D
from split import split_motor, split_wustl, split_wustl_multi


# ======================================================
# UTILS
# ======================================================
def build_exp_name(args):
    parts = [args.exp_name, args.mode, f"sub-{args.subject}"]

    if args.mode == "motor":
        parts.append(args.train_dataset)

        if args.train_dataset == "yuanyuan":
            parts.append(f"baseline")

        if args.train_dataset == "laura":
            parts.append(f"fnirs")

        if args.train_dataset == "fmri":
            parts.append(f"fmri{args.fmri_subjects}")

    return "_".join(parts)


# ======================================================
# TRAIN / EVAL
# ======================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for x, y, modality in loader:
        x, y, modality = x.to(device), y.to(device), modality.to(device)

        out = model(x, modality)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y, modality in loader:
            x, y, modality = x.to(device), y.to(device), modality.to(device)

            out = model(x, modality)
            preds = torch.argmax(out, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return f1_score(all_labels, all_preds, average="macro")


# ======================================================
# SPLITS
# ======================================================
def get_split(args):

    if args.mode == "motor":
        return split_motor(
            None,
            args.data_path,
            test_subjects_list=[args.subject],
            exclude_subjects=args.exclude,
            train_dataset=args.train_dataset,
            fmri_subjects=args.fmri_subjects
        )

    elif args.mode == "wustl":
        return split_wustl(
            None,
            args.data_path,
            test_subjects_list=[args.subject],
            val_subjects_list=None
        )

    elif args.mode == "wustl_multi":
        return split_wustl_multi(
            None,
            args.data_path,
            test_subjects_list=[args.subject]
        )


# ======================================================
# SAMPLERS
# ======================================================
def build_sampler(args, dataset):

    # MOTOR → modality balancing
    if args.mode == "motor" and args.train_dataset == "fmri":
        print("→ Using ProportionalModalityBatchSampler")

        return ProportionalModalityBatchSampler(
            dataset.all_modalities,
            batch_size=args.batch_size,
            fnirs_ratio=args.fnirs_ratio
        )

    # WUSTL → class balancing
    if args.mode in ["wustl", "wustl_multi"]:
        print("→ Using WeightedRandomSampler")

        labels = [dataset[i][1] for i in range(len(dataset))]
        labels = np.array(labels)

        class_counts = np.bincount(labels)
        weights = 1.0 / (class_counts + 1e-8)

        sample_weights = weights[labels]
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    return None


# ======================================================
# CLASS WEIGHTS
# ======================================================
def get_class_weights(dataset, args, device):

    if args.class_weights is not None:
        print("→ Using MANUAL class weights:", args.class_weights)
        return torch.tensor(args.class_weights, dtype=torch.float32).to(device)

    if args.class_weight_mode == "auto":
        print("→ Using AUTO class weights")

        labels = [dataset[i][1] for i in range(len(dataset))]
        labels = np.array(labels)

        class_counts = np.bincount(labels)
        weights = 1.0 / (class_counts + 1e-8)
        weights = weights / weights.sum()

        print("Class counts:", class_counts)
        print("Class weights:", weights)

        return torch.tensor(weights, dtype=torch.float32).to(device)

    return None


# ======================================================
# MODEL
# ======================================================
def get_model(args):
    if args.mode == "motor":
        return CNN2D(in_channels=104, num_classes=2)
    elif args.mode == "wustl":
        return CNN2D(in_channels=371, num_classes=2)
    elif args.mode == "wustl_multi":
        return CNN2D(in_channels=371, num_classes=5)


# ======================================================
# MAIN
# ======================================================
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = build_exp_name(args)
    os.makedirs("checkpoints", exist_ok=True)

    print(f"\n🚀 Experiment: {exp_name}\n")

    # -------- SPLIT --------
    train_csv, val_csv, test_csv = get_split(args)

    # -------- DATASET --------
    train_dataset = fNIRSPreloadDataset(train_csv, chromo="both")
    val_dataset   = fNIRSPreloadDataset(val_csv, chromo="both")
    test_dataset  = fNIRSPreloadDataset(test_csv, mode="test", chromo="both")

    # -------- SAMPLER --------
    sampler = build_sampler(args, train_dataset)

    if sampler is not None:
        train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # -------- MODEL --------
    model = get_model(args).to(device)

    # -------- LOSS --------
    weights = get_class_weights(train_dataset, args, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_f1 = 0

    # -------- TRAIN --------
    for epoch in range(args.epochs):

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_f1 = evaluate(model, val_loader, device)

        print(f"[{epoch:03d}] Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join("checkpoints", f"{exp_name}_best.pth")
            torch.save(model.state_dict(), save_path)

    print("\n Final Test F1:", evaluate(model, test_loader, device))


# ======================================================
# CLI
# ======================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", required=True,
                        choices=["motor", "wustl", "wustl_multi"])

    parser.add_argument("--exp_name", type=str, default="exp")

    parser.add_argument("--data_path", required=True)
    parser.add_argument("--subject", required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)

    # motor-specific
    parser.add_argument("--train_dataset", default="yuanyuan")
    parser.add_argument("--fmri_subjects", type=int, default=24)
    parser.add_argument("--fnirs_ratio", type=float, default=0.5)

    parser.add_argument("--exclude", nargs="*", default=None)

    # class weights
    parser.add_argument("--class_weight_mode", choices=["none", "auto"], default="none")
    parser.add_argument("--class_weights", nargs="+", type=float, default=None)

    args = parser.parse_args()

    main(args)