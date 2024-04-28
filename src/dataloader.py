from datasets import load_dataset
from torch.utils.data import DataLoader

def load_squad(split="train"):
    dataset = load_dataset("rajpurkar/squad_v2", split=split)
    return dataset
    
def get_contexts():
    dataset = load_squad()
    tokens = 0
    contexts = []
    for i in range(len(dataset)):
        contexts.append(dataset[i]['context'])
        tokens += len(dataset[i]['context'].split())
    print(tokens)
    print(len(contexts))
    return contexts

class SQUADataset:
    def __init__(self, split="train"):
        self.dataset = load_squad(split)
        self.max_length = 384
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", padding="max_length", truncation=True)
        self.dataset = self.dataset.map(
            lambda e: self.tokenizer(e["question"], e["context"], max_length=self.max_length, padding="max_length", truncation=True),
            batched=True,
        )
        self.dataset.set_format(type="torch", columns=['id', 'question', 'context', 'input_ids', 'attention_mask', 'token_type_ids'])
        self.dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


get_questions()