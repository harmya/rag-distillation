from transformers import BertModel, BertTokenizer

class Teacher:
    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def model():
        return self.model