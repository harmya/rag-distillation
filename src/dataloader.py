from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def load_squad(split="train"):
    dataset = load_dataset("squad", split=split)
    return dataset
    
class SQUADataset:
    def __init__(self, split="train"):
        self.dataset = load_squad(split)
        self.max_length = 384
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", padding="max_length", truncation=True)
        self.dataset = self.dataset.map(
            lambda e: self.tokenizer(e["question"], e["context"], max_length=self.max_length, padding="max_length", truncation=True),
            batched=True,
        )
        self.dataset.set_format(type="torch", columns=['input_ids', 'token_type_ids', 'attention_mask'])
        self.dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


sd = SQUADataset()
print(sd.tokenizer.decode(sd[0]['input_ids']))