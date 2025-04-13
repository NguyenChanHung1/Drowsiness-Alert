import torch 
import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
import yaml
import torch.nn as nn
import os
import wandb
import sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.CNN_LSTM import CNN_LSTM
from train_utils import load_data, validate_config, validate_config_CNN_LSTM, save_model, load_model

def epoch(model: nn.Module, train_loader, val_loader, loss_function, optimizer, device):    
    train_loss_epoch = 0
    eval_loss_epoch = 0
    model.train()
    
    for i, (data, label) in enumerate(tqdm(train_loader)):
        data, label = data.float().to(device), label.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        
        loss = loss_function(outputs, label)
        loss.backward()
        
        optimizer.step()
        train_loss_epoch += loss.item()
    train_loss_epoch /= (i + 1)
    
    model.eval()
    with torch.no_grad():
        num_val = 0
        num_correct = 0
        print("Attempting validation...")
        for j, (valid_data, valid_label) in enumerate(tqdm(val_loader)):
            valid_data, valid_label = valid_data.float().to(device), valid_label.to(device)
            eval_output = model(valid_data)
            eval_loss = loss_function(eval_output, valid_label)
            eval_loss_epoch += eval_loss.item()
            num_val += valid_label.size(0)
            _, pred = torch.max(eval_output, 1)
            num_correct += torch.sum(pred == valid_label)
    eval_loss_epoch /= (j + 1)
    eval_acc = num_correct / num_val
    return model, optimizer, train_loss_epoch, eval_loss_epoch, eval_acc

def train_CNN_LSTM(train_data, val_data, epochs, device, model_info):
    # Parse hyperparameters
    backbone = model_info["backbone"]
    hidden_size = model_info["hidden_size"]
    num_lstm_layers = model_info["num_lstm_layers"]
    learning_rate = float(model_info["lr0"])
    model = CNN_LSTM(backbone, hidden_size, num_lstm_layers).to(device)

    # Train model
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    learning_rate_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.4)
    loss_function = nn.CrossEntropyLoss()

    # Log in to wandb
    wandb.login(
        key = "7d96822b87524b867791ff3de2483dac5f7097fc",
    )
    wandb.init(project="CNN_LSTM")

    # Training loop
    max_val_acc = 0
    for ep in (range(epochs)):
        print(f"Start epoch {ep+1}:")
        this_ep_lr = learning_rate_scheduler.get_last_lr()[0]
        model, optimizer, train_loss_ep, val_loss_ep, val_acc_ep = epoch(model, train_data, val_data, loss_function, optimizer, device)
        if val_acc_ep > max_val_acc:
            save_model(model, optimizer, path="ml/weights/CNN_LSTM/best.pt")
            max_val_acc = val_acc_ep
        save_model(model, optimizer, path="ml/weights/CNN_LSTM/last.pt")
        learning_rate_scheduler.step()
        wandb.log({
            "Train loss": train_loss_ep,
            "Val loss": val_loss_ep,
            "Val accuracy": val_acc_ep,
            "Learning rate": this_ep_lr
        })
        print(f"Epoch {ep+1} finished. Train loss: {train_loss_ep}, Val loss: {val_loss_ep}, Val_accuracy: {val_acc_ep}.")

def train(annotation: str, config: str):
    config_obj = yaml.load(open(config, 'r'), yaml.SafeLoader)
    validate_config(config_obj)
    model = config_obj["model_info"]["model_type"]
    if model not in ["CNN_LSTM"]:
        raise NotImplementedError
    train_hyperparams = config_obj["train_hyp"]

    batch_size = train_hyperparams["batch_size"]
    num_workers = train_hyperparams["num_workers"]
    epochs = train_hyperparams["epochs"]
    device = "cuda" if train_hyperparams["gpu"] and torch.cuda.is_available() else "cpu"

    train_data = load_data(annotation, "train", batch_size, num_workers)
    val_data = load_data(annotation, "valid", batch_size, num_workers)
    # return 
    
    if model == "CNN_LSTM":
        validate_config_CNN_LSTM(config_obj["model_info"])
        train_CNN_LSTM(train_data, val_data, epochs, device, config_obj["model_info"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', help="Dataset annotation path")
    parser.add_argument('--config', help="Model config .txt file")
    args = parser.parse_args()
    train(args.ann, args.config)

if __name__ == '__main__':
    main()