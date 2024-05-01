import json
from transformers import pipeline, set_seed
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset

# generator = pipeline('text-generation', model='/scratch/gilbreth/mangla/exps-rag/checkpoint-26500', tokenizer="openai-community/gpt2", device=0)
generator = pipeline('text-generation', model='facebook/opt-125m', tokenizer="facebook/opt-125m", device=0)
generator.tokenizer.padding_side = "left"
generator.tokenizer.pad_token_id = generator.model.config.eos_token_id

dev = "/scratch/gilbreth/mangla/rag-distillation/data/dev.json"
outfile = "/scratch/gilbreth/mangla/rag-distillation/data/vanilla-OPT.json"

class Squad(Dataset):
    def __getitem__(self, idx):
        q = f'Question: {qs[idx]}\nAnswer:'
        return q

    def __len__(self):
        return len(qs)

qs = []
contexts = []
q_ids = {}

responses = {}
with open(dev, 'r') as file:
    data = json.load(file)
    for item in tqdm(data['data']):
        for paragraph in item['paragraphs']:
            for qa in paragraph['qas']:
                prompt = qa['question']
                q_id = qa['id']
                q = f'Question: {prompt}\nAnswer:'
                q_ids[q] = q_id
                qs.append(prompt)
                contexts.append(paragraph['context'])

dataset = Squad()

i = 0
for out in tqdm(generator(dataset, max_new_tokens=30, batch_size=8), total=len(dataset)):
    prompt = dataset[i]
    generated_text = out[0]['generated_text'][len(prompt):]
    generated_text = generated_text.replace("\n", " ").strip()
    responses[q_ids[prompt]] = generated_text
    i += 1

with open(outfile, 'w') as out_file:
    json.dump(responses, out_file, indent=4)