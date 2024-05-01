import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import json
from transformers import Trainer, TrainingArguments
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling

device = 0

class DistillDataGPT(Dataset):
    def __init__(self, dataset, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        curr = self.dataset[idx]
        complete_str = f'Question: {curr["question"]}\nAnswer: {curr["answer"]}'

        encoding = self.tokenizer.encode_plus(
            complete_str,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            padding='max_length',   
            max_length=128,
            truncation=True
        )
        return encoding['input_ids']


# 
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

data_list = []
with open('../../data/squad_teacher.jsonl', 'r') as file:
    for line in file:
        data_instance = json.loads(line)
        data_list.append(data_instance)

distill_dataset = DistillDataGPT(data_list, tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

training_args = TrainingArguments(
    output_dir="/scratch/gilbreth/mangla/exps-squad-gpt",
    overwrite_output_dir=False,
    per_device_train_batch_size=64,
    num_train_epochs=50,
    logging_dir='/scratch/gilbreth/mangla/exps-log',
    logging_steps=20,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=distill_dataset
)
    
trainer.train()
trainer.save_model()
