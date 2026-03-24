import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import pickle

from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score

from datasets_v02 import fNIRSPreloadDataset
from model import Bold_Hybrid
from split import split_motor, split_wustl, split_wustl_multi


# ======================================================
# EXP NAME
# ======================================================
def build_exp_name(args):
    return f"{args.exp_name}_{args.mode}_{args.freeze_mode}_sub-{args.subject}"


# ======================================================
# FREEZING STRATEGY
# ======================================================
def apply_freezing(model, args):

    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # always train projections
    for p in model.project_hbo.parameters():
        p.requires_grad = True

    for p in model.project_hbr.parameters():
        p.requires_grad = True

    # classifier
    if args.freeze_mode in ["proj_cls", "proj_cls_tr"]:
        for p in model.classification.parameters():
            p.requires_grad = True

    # transformer last layer
    if args.freeze_mode == "proj_cls_tr":
        for p in model.transformer.layers[-1].parameters():
            p.requires_grad = True


# ======================================================
# MODEL
# ======================================================
def build_model(args, device):

    model = Bold_Hybrid().to(device)

    print("Loading checkpoint:", args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # set classifier
    num_classes = 2 if args.mode != "wustl_multi" else 5
    model.classification = nn.Linear(model.embedding_dim, num_classes).to(device)

    apply_freezing(model, args)

    return model


# ======================================================
# TRAIN / EVAL
# ======================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        x = x.permute(0, 2, 1, 3)

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            x = x.permute(0, 2, 1, 3)

            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()
            preds = torch.argmax(out, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average="macro")
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    return total_loss / len(loader), acc, f1


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
            train_dataset="yuanyuan",
            fmri_subjects=24
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
# SAMPLER
# ======================================================
def build_sampler(args, dataset):

    if args.mode in ["wustl", "wustl_multi"]:
        labels = [dataset[i][1] for i in range(len(dataset))]
        labels = np.array(labels)

        class_counts = np.bincount(labels)
        weights = 1.0 / (class_counts + 1e-8)

        sample_weights = torch.tensor(weights[labels], dtype=torch.float32)

        print("Using WeightedRandomSampler")
        return WeightedRandomSampler(sample_weights, len(sample_weights))

    return None


# ======================================================
# MAIN
# ======================================================
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = build_exp_name(args)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print(f"\n Finetuning: {exp_name}\n")

    # -------- split --------
    train_df, val_df, test_df = get_split(args)

    # -------- dataset --------
    train_dataset = fNIRSPreloadDataset(train_df, chromo="both")
    val_dataset   = fNIRSPreloadDataset(val_df, chromo="both")
    test_dataset  = fNIRSPreloadDataset(test_df, mode="test", chromo="both")

    sampler = build_sampler(args, train_dataset)

    if sampler:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # -------- model --------
    model = build_model(args, device)

    # -------- optimizer --------
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    best_state = None
    patience_counter = 0

    history = {"train_f1": [], "val_f1": []}

    # -------- train --------
    for epoch in range(args.epochs):

        train_one_epoch(model, train_loader, criterion, optimizer, device)

        _, train_acc, train_f1 = evaluate(model, train_loader, criterion, device)
        _, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"[{epoch:03d}] Train F1: {train_f1:.3f} | Val F1: {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            torch.save(best_state, f"checkpoints/{exp_name}.pth")

        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping")
            break

        scheduler.step(val_f1)

    # -------- test --------
    print("\nLoading best model...")
    model.load_state_dict(best_state)

    _, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)

    print("\n FINAL TEST F1:", test_f1)

    with open(f"results/{exp_name}.pkl", "wb") as f:
        pickle.dump({
            "history": history,
            "test_f1": test_f1
        }, f)


# ======================================================
# CLI
# ======================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", required=True,
                        choices=["motor", "wustl", "wustl_multi"])

    parser.add_argument("--exp_name", default="ft")
    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--data_path", required=True)
    parser.add_argument("--subject", required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--freeze_mode",
                        choices=["proj", "proj_cls", "proj_cls_tr"],
                        default="proj_cls_tr")

    parser.add_argument("--exclude", nargs="*", default=None)

    args = parser.parse_args()

    main(args)