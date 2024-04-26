from transformers import BertForQuestionAnswering, BertTokenizer
import torch

model_path = 'student_epoch_9.pt'

def load_model(model_path):
    state_dict = torch.load(model_path)
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
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

question = "What is the capital of France?"
context = "The capital of France is Paris."

model = load_model(model_path)
answer = answer_question(question, context, model)