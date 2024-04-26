from transformers import BertForQuestionAnswering, BertTokenizer, BertConfig

class Teacher:
    def __init__(self, vocab_size=30522):
        self.model = BertForQuestionAnswering.from_pretrained('google-bert/bert-large-uncased-whole-word-masking-finetuned-squad')
        print(f"Total parameters: {self.model.num_parameters()}")

    def model():
        return self.model
