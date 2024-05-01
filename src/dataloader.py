from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class SquadDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        return self.data[idx]