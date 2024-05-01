# Leveraging RAG-Assisted Teacher Models in Knowledge-Distillation for Enhanced Domain-Specific Question Answering

Code for CS577 Natural Language Processing project. The `src` directory contains the source code for the majority of our experiments. We also experimented with available KD frameworks like [MiniLLM](https://github.com/microsoft/LMOps/tree/main/minillm) and [Distilling Step-by-Step](https://github.com/google-research/distilling-step-by-step). All the models and tokenizers are pulled from the [huggingface](https://pypi.org/project/transformers/) hub.

## Data Preprocessing
We utilize the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/) as our primary benchmark and distillation dataset. Our preprocessing script `preprocess.py` converts the raw SQuAD data into a uniform format to be passed into the model. The two eval scripts `eval.py` and `eval2.py` generate predictions for `dev` set, and we use SQuAD's official evaluation script (`squad_eval.py` in our code) to compute metrics over the gold outputs and our predictions. 

## Knowledge Bases
We use [Cohere's wikipedia embeddings](https://huggingface.co/datasets/Cohere/wikipedia-22-12-simple-embeddings) as our generic knowledge base, the `knowledge.py` script generates context for the teacher models using Cohere's API.

## Knowledge Distillation Training
We have different scripts corresponding to different experiments we ran throughout the project. `finetune.py` fine-tunes a given model from Hugging Face hub on a given dataset, and `distill.py` distills knowledge using the logits from the teacher model. Additionally, `student.py` and `teacher.py` contain definitions of one iteration of experiments, and `dataloader.py` contains the dataloader used in training.

By Harmya, Sarthak
