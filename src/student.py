from transformers import BertConfig, BertModel

class Student:
    def __init__(self, num_hidden_layers=6, num_attention_heads=12):
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.config.num_hidden_layers = num_hidden_layers
        self.config.num_attention_heads = num_attention_heads
        self.model = BertModel(self.config)


    def model():
        return self.model