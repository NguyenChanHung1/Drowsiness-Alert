from torch.utils.data import DataLoader
import torch
from dataset import DrowsinessData

def load_data(annotation_path: str, split: str, batch_size: int, num_workers: int):
    dataset = DrowsinessData(annotation=annotation_path, split=split) # 90,3,224,224
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return dataloader

def validate_config(config_obj): 
    required_keys = ["train_hyp", "model_info"]
    required_train_hyp_keys = ["batch_size", "epochs", "num_workers"]
    missing_keys = [key for key in required_keys if key not in config_obj] + [tkey for tkey in required_train_hyp_keys if tkey not in config_obj["train_hyp"]] 

    if missing_keys:
        raise KeyError("Missing keys detected in yaml config file:", missing_keys)

    if "model_type" not in config_obj["model_info"]:
        raise KeyError("Missing key in :model_info:model_type")

def validate_config_CNN_LSTM(config_obj):
    required_keys = ["backbone", "hidden_size", "num_lstm_layers", "lr0"]
    missings = [key for key in required_keys if key not in config_obj]
    if missings:
        raise KeyError("Missing keys detected in yaml config file:", missings)

    print("No error when loading config file. Attempting training...")

def save_model(model, optimizer, path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer