import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import wandb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.STGCN import STGCN, TwoStreamSTGCN  
from dataset import DrowsinessData  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
annotation_file = "ml/dataset/data.csv" # path to the CSV file
save_dir = "ml/weights/STGCN"
os.makedirs(save_dir, exist_ok=True)

batch_size = 16
num_epochs = 200
learning_rate = 1e-4
weight_decay = 5e-6

def main(number_frames, deeper, twostream):
    # Load data
    train_dataset = DrowsinessData(annotation_file, T_frame=number_frames, split="train")
    val_dataset = DrowsinessData(annotation_file, T_frame=number_frames, split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    if twostream:
        model = TwoStreamSTGCN(in_channel=2, num_class=2, T_frame=number_frames)
        model.to(device)
    else:
        model = STGCN(in_channels=2, num_class=2, T_frame=number_frames, deep=deeper, device=device)
        model.to(device)

    # Loss, optim, scheduler
    criterion = nn.BCELoss()  # Binary
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)

    best_val_acc = 0.0
    best_train_acc = 0.0

    wandb.login(key="7d96822b87524b867791ff3de2483dac5f7097fc")
    wandb.init(project="Face ST-GCN")

    # Train loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")
        for landmarks_seq, labels in pbar:
            landmarks_seq = landmarks_seq.permute(0, 3, 1, 2).to(device)  # [B, 2, 40, 30] 40 frames, 30 keypoints
            labels = labels.to(device).float().unsqueeze(1)  # [N, 1]

            optimizer.zero_grad()
            outputs = model(landmarks_seq)  
            # print("ao put", outputs)
            if outputs.shape[1] == 2:
                outputs = outputs[:, 1].unsqueeze(1)  # drowsy class prob

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = (outputs > 0.5).long()
            correct_train += (preds.squeeze() == labels.squeeze().long()).sum().item()
            total_train += labels.size(0)

            pbar.set_postfix({"loss": train_loss / (total_train / batch_size), "acc": correct_train / total_train})

        train_acc = correct_train / total_train

        scheduler.step()

        # Valid
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for landmarks_seq, labels in tqdm(val_loader, desc="Validating"):
                landmarks_seq = landmarks_seq.permute(0, 3, 1, 2).to(device)  # [B,2,40,30]
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

        wandb.log({"Train loss": train_loss/len(train_loader), "Train accuracy": train_acc, "Val loss": val_loss/len(val_loader), "Val accuracy": val_acc})

        layers_tag = ""
        if deeper:
            layers_tag += "6layers"
        elif deeper==False and istwostream==False:
            layers_tag += "3layers"
        else:
            layers_tag += "twostream"

        # Save model if best
        if val_acc >= best_val_acc or (train_acc >= best_train_acc and val_acc >= best_val_acc):
            best_train_acc = train_acc
            best_val_acc = val_acc
            checkpoint_path = os.path.join(save_dir, f"best_t{number_frames}_{layers_tag}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"Saved Best Model to {checkpoint_path}")

        last_path = os.path.join(save_dir, f"last_model_{number_frames}_{layers_tag}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, last_path)
        print(f"Saved Last Model to {last_path}")

    print("Training Complete!")

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nf', '--number-frames', help="The number of frames for each sample", default=40)
    parser.add_argument('-d', '--deeper', help="If deeper=True, the number of STGCN layers is 6, otherwise it's 3", default=False)
    parser.add_argument('--twostream', help="Switch to two-stream mode", default=0)
    args = parser.parse_args()
    istwostream = True if int(args.twostream) > 0 else False
    main(int(args.number_frames), int(args.deeper), istwostream)