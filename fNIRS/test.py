from torch.utils.data import DataLoader
import pickle
from datasets_v02 import fNIRSPreloadDataset
import torch
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model import ImprovedTransformer, BoldT, BoldT_Conv, ImprovedTransformerDual, Bold_Hybrid
import warnings
import matplotlib.pyplot as plt
from utils import create_train_test_files, create_train_test_segments
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Bold_Hybrid().to(device)
checkpoint = "/home/checkpoints/model_fmri_hybrid_opt.pth"

model.load_state_dict(torch.load(checkpoint))


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels, _ in test_loader:
            data, labels = data.to(device), labels.to(device)
            data = data.permute(0, 2, 1, 3)

            logits_21 = model(data)
            outputs = logits_21[:, [0, 1]]
            
            # outputs = logits_21

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="micro")

    return total_loss / len(test_loader), accuracy, f1


batch_size = 32
# Load datasets
base_dir = "/home"
dataset_path = os.path.join(base_dir, "data/BS_Laura")
preprocessed_path = os.path.join(base_dir, "data/combined_fnirs_fh_final")

# test_subjects_list = ['sub-577', 'sub-581', 'sub-586',  
#                 'sub-613', 'sub-619', 'sub-633', 'sub-568', 
#                 'sub-580', 'sub-583', 'sub-587', 'sub-592',  
#                 'sub-618', 'sub-621', 'sub-638', 'sub-640']
test_subjects_list = ['sub-177', 'sub-182', 'sub-185', 'sub-633', 'sub-176',
                        'sub-580', 'sub-583', 'sub-586', 'sub-618', 'sub-640',
                        'sub-568', 'sub-621']




exclude_subjects = ['sub-547', 'sub-639', 'sub-588', 'sub-171', 'sub-174', 'sub-184']
criterion = nn.CrossEntropyLoss()
results = {}

for subs in test_subjects_list:

    print("Evaluating subject:", subs)

    _, test_csv_path = create_train_test_segments(
        None, preprocessed_path, 
        test_subjects_list=[subs],   # ✅ only this subject
        exclude_subjects=exclude_subjects
    )

    test_dataset = fNIRSPreloadDataset(test_csv_path, mode="test", chromo='both')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_loss, test_accuracy, test_f1 = evaluate_model(model, test_loader, criterion, device)

    results[subs] = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1
    }

    print(f"{subs} | Loss: {test_loss:.4f} | Acc: {test_accuracy:.3f} | F1: {test_f1:.3f}")

with open("/home/results/zero_shot_final.pkl", "wb") as f:
    pickle.dump(results, f)
