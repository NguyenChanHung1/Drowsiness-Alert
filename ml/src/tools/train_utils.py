from torch.utils.data import DataLoader
import torch
from dataset import DrowsinessData

def load_data(annotation_path: str, split: str, batch_size: int, num_workers: int):
    dataset = DrowsinessData(annotation=annotation_path, split=split) # 
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return dataloader

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