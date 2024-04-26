from transformers import BertForQuestionAnswering, BertTokenizer

class Teacher:
    def __init__(self):
        self.model = BertForQuestionAnswering.from_pretrained('bert-small-finetuned-squad')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print("Teacher model initialized.")
        print(f"Total parameters: {self.model.num_parameters()}")

    def model():
        return self.model