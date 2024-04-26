from transformers import BertConfig, BertForQuestionAnswering

class Student:
    def __init__(self, num_hidden_layers=6, num_attention_heads=6):
        self.config = BertConfig(
            hidden_size=num_attention_heads * 64,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=512,
        )
        self.config.num_hidden_layers = num_hidden_layers
        self.config.num_attention_heads = num_attention_heads
        self.model = BertForQuestionAnswering(self.config)
        print(f"Student model initialized with {num_hidden_layers} hidden layers and {num_attention_heads} attention heads.")
        print(f"Total parameters: {self.model.num_parameters()}")

    def model():
        return self.model