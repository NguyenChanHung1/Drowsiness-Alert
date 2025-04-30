import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === import your modules ===
from model.STGCN import STGCN  # assuming your STGCN is in model.py
from dataset import DrowsinessData  # assuming your dataset is in dataset.py

# === Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
annotation_file = "ml/dataset/data.csv" # path to the CSV file
save_dir = "ml/weights/STGCN"
os.makedirs(save_dir, exist_ok=True)

batch_size = 16
num_epochs = 1000
learning_rate = 1e-3
weight_decay = 1e-4

# === Load Dataset ===
train_dataset = DrowsinessData(annotation_file, split="train")
val_dataset = DrowsinessData(annotation_file, split="val")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# === Initialize Model, Loss, Optimizer ===
model = STGCN(in_channels=2, num_class=2, device=device)
model.to(device)

criterion = nn.BCELoss()  # Since output is sigmoid for binary
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

best_val_acc = 0.0

wandb.login(key="7d96822b87524b867791ff3de2483dac5f7097fc")
wandb.init(project="Face ST-GCN")

# === Training and Validation Loop ===
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")
    for landmarks_seq, labels in pbar:
        landmarks_seq = landmarks_seq.permute(0, 3, 1, 2).to(device)  # [N, 2, T, 68]
        labels = labels.to(device).float().unsqueeze(1)  # [N, 1]

        optimizer.zero_grad()
        outputs = model(landmarks_seq)  # [N, 1] or [N, 2] depending on model output
        if outputs.shape[1] == 2:
            outputs = outputs[:, 1].unsqueeze(1)  # take "drowsy" class probability

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = (outputs > 0.5).long()
        correct_train += (preds.squeeze() == labels.squeeze().long()).sum().item()
        total_train += labels.size(0)

        pbar.set_postfix({"loss": train_loss / (total_train / batch_size), "acc": correct_train / total_train})

    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for landmarks_seq, labels in tqdm(val_loader, desc="Validating"):
            landmarks_seq = landmarks_seq.permute(0, 3, 1, 2).to(device)  # [B,2,90,68]
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(landmarks_seq)
            if outputs.shape[1] == 2:
                outputs = outputs[:, 1].unsqueeze(1)

            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = (outputs > 0.5).long()
            correct_val += (preds.squeeze() == labels.squeeze().long()).sum().item()
            total_val += labels.size(0)

    val_acc = correct_val / total_val
    print(f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, Val Acc {val_acc:.4f}")

    wandb.log({"Train loss": train_loss/len(train_loader), "Val loss": val_loss/len(val_loader), "Val accuracy": val_acc})

    # Save model if best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        checkpoint_path = os.path.join(save_dir, "best_model_epoch.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, checkpoint_path)
        print(f"Saved Best Model to {checkpoint_path}")

print("Training Complete!")