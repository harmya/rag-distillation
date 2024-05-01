# Leveraging RAG-Assisted Teacher Models in Knowledge-Distillation for Enhanced Domain-Specific Question Answering

Code for CS577 Natural Language Processing project. The `src` directory contains the source code for the majority of our experiments. We also experimented with available KD frameworks like [MiniLLM](https://github.com/microsoft/LMOps/tree/main/minillm) and [Distilling Step-by-Step](https://github.com/google-research/distilling-step-by-step). All the models and tokenizers are pulled from the [huggingface](https://pypi.org/project/transformers/) hub.

# Data Preprocessing
We utilize the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/) as our primary benchmark and distillation dataset. Our two eval scripts `eval.py` and `eval2.py` generate predictions for `dev` set, and we use SQuAD's official evaluation script (`squad_eval.py` in our code) to compute metrics over the gold outputs and our predictions. 



By Harmya, Sarthak
