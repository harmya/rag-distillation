import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-large-squad2"

data_file = "../../data/cohere_train.jsonl"
teacher = pipeline('question-answering', model=model_name, tokenizer=model_name, device=0)

class SquadDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        return self.data[idx]

with open(data_file, 'r') as file:
    data_lines = file.readlines()

qa_data = []
final_data = []
for line in data_lines:
    data_instance = json.loads(line)
    qa_instance = {"question": data_instance.get("question", ""), "context": data_instance.get("context", "")}
    qa_data.append(qa_instance)

dataset = SquadDataset(qa_data)

dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

curr = 0
for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    out = teacher(batch)
    for j in range(len(out)):
        final_instance = {"question": batch["question"][j], "context": batch["context"][j], "answer": out[j]["answer"]}
        final_data.append(final_instance)
        curr += 1

with open('../../data/cohere_teacher.jsonl', 'w') as output_file:
    for item in final_data:
        output_file.write(json.dumps(item) + '\n')
