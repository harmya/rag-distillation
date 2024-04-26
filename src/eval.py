from transformers import BertForQuestionAnswering, BertTokenizer
import torch
from student import Student

model_path = '../models/student_epoch_9.pt'

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
    inputs = tokenizer(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer

    
question = "which nfl team represented the afc at super bowl 50?"
context = "super bowl 50 was an american football game to determine the champion of the national football league (nfl) for the 2015 season. the american football conference (afc) champion denver broncos defeated the national football conference (nfc) champion carolina panthers 24–10 to earn their third super bowl title. the game was played on february 7, 2016, at levi's stadium in the san francisco bay area at santa clara, california."

model = load_model(model_path)
answer = answer_question(question, context, model)
print(answer)