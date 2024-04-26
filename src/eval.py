from transformers import BertForQuestionAnswering, BertTokenizer
import torch
from student import Student
from dataloader import SQUADataset
import json

model_path = '../models/student_epoch_10.pt'

def load_model(model_path):
    state_dict = torch.load(model_path)
    for key in list(state_dict.keys()):
        if 'module.' in key:
            state_dict[key.replace('module.', '')] = state_dict.pop(key)
    student = Student()
    model = student.model
    model.load_state_dict(state_dict)
    model.eval()
    return model


def answer_question(question, context, model):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(question, context, return_tensors="pt", max_length=384, padding="max_length", truncation=True)
    input_ids = inputs["input_ids"].tolist()[0]
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer

model = load_model(model_path)
sd_val = SQUADataset(split="validation")

eval_dict = {}
print(len(sd_val))

for i in range(len(sd_val)):
    question = sd_val[i]['question']
    context = sd_val[i]['context']
    answer = answer_question(question, context, model)
    dict_id = sd_val[i]['id']
    eval_dict[dict_id] = answer


with open('eval_dict.json', 'w') as f:
    json.dump(eval_dict, f)

print("Evaluation complete. Results saved to eval_dict.json")


