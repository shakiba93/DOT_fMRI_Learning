from torch.utils.data import DataLoader
import pickle
from fMRI_Basic_ML.datasets import HCPTrialDataset
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from fMRI_Basic_ML.model import Bold_Hybrid
import warnings
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

print("hello hydra")
print(f"job ID: {os.getenv('SLURM_JOB_ID')}")
print(f"array job ID: {os.getenv('SLURM_ARRAY_JOB_ID')}")
print(f"array task ID: {os.getenv('SLURM_ARRAY_TASK_ID')}")
print(f"CUDA available: {torch.cuda.is_available()}")

# -------------------------
# TRAIN FUNCTION (unchanged except grad clip)
# -------------------------
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)

        outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (very important for transformers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# -------------------------
# EVALUATION FUNCTION (unchanged)
# -------------------------
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_preds, average="macro")  # <-- better for imbalance

    return total_loss / len(loader), accuracy, f1



# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    # Hyperparameters
    num_epochs = 200
    learning_rate = 1e-4
    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = "/home"
    dataset_path = os.path.join(base_dir, "HCP_1200_all")

    # ------------------------------------------------
    # LOAD DATASETS
    # ------------------------------------------------
    train_dataset = HCPTrialDataset(os.path.join(dataset_path, "trials_train_15"))
    val_dataset   = HCPTrialDataset(os.path.join(dataset_path, "trials_val_15"))
    test_dataset  = HCPTrialDataset(os.path.join(dataset_path, "trials_test_15"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    print("Datasets loaded: train / val / test")

    # ------------------------------------------------
    # MODEL + LOSS + OPTIMIZER (tuned)
    # ------------------------------------------------
    model = Bold_Hybrid().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.05      # <-- increased from 0.01
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.5
    )

    # ------------------------------------------------
    # METRICS STORAGE
    # ------------------------------------------------
    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []
    train_f1s, val_f1s, test_f1s = [], [], []

    # ------------------------------------------------
    # EARLY STOPPING (clean version)
    # ------------------------------------------------
    best_val_f1 = 0.0
    patience = 10
    no_improve = 0
    best_state = None

    # ------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------
    for epoch in range(num_epochs):

        train_loss = train_model(model, train_loader, criterion, optimizer, device)

        train_loss_eval, train_acc, train_f1 = evaluate_model(
            model, train_loader, criterion, device
        )

        val_loss, val_acc, val_f1 = evaluate_model(
            model, val_loader, criterion, device
        )

        test_loss, test_acc, test_f1 = evaluate_model(
            model, test_loader, criterion, device
        )

        # store
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)

        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        test_f1s.append(test_f1)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f} || "
            f"Train Acc: {train_acc:.3f} (F1={train_f1:.3f}) | "
            f"Val Acc: {val_acc:.3f} (F1={val_f1:.3f}) | "
            f"Test Acc: {test_acc:.3f} (F1={test_f1:.3f})"
        )


        # Early stopping on VAL LOSS (more stable than F1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
            no_improve = 0
            torch.save(best_state, "/home/checkpoints/fmri_hybrid_opt.pth")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        # Step scheduler AFTER computing val_loss
        scheduler.step(val_loss)

    # ------------------------------------------------
    # Load best model before final test
    # ------------------------------------------------
    print("\nLoading best model for final test...")
    model.load_state_dict(best_state)

    test_loss, test_acc, test_f1 = evaluate_model(
        model, test_loader, criterion, device
    )

    print("\nFINAL TEST RESULTS")
    print("Test Accuracy:", test_acc)
    print("Test F1:", test_f1)

    # ------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------
    res = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "test_loss": test_losses,

        "train_accuracy": train_accuracies,
        "val_accuracy": val_accuracies,
        "test_accuracy": test_accuracies,

        "train_f1": train_f1s,
        "val_f1": val_f1s,
        "test_f1": test_f1s,

        "best_val_f1": best_val_f1,
        "final_test_acc": test_acc,
        "final_test_f1": test_f1,
    } 


    with open("/home/results/res_fmri_hybrid_opt.pkl", "wb") as f:
        pickle.dump(res, f)

    torch.save(model.state_dict(), "/home/checkpoints/model_fmri_hybrid_opt.pth")
    print("Training finished and results saved.")
