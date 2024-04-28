import torch
import cohere
import os
import dotenv
import dataloader
from torch.utils.data import DataLoader
from datasets import load_dataset

dotenv.load_dotenv()


co = cohere.Client(f"{os.getenv('COHERE_PROD')}")

dataset = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True)

docs = []
loaded = 0

for doc in dataset:
    docs.append(doc)
    loaded += 1
    print(f"Loaded {loaded} documents", end="\r")


doc_embeddings = torch.load(doc_embeddings).to('cuda')

rag_outputs = {}
queries = dataloader.get_questions()

for query_id, query in queries.items():
    response = co.embed(texts=[query], model='multilingual-22-12')
    query_embedding = torch.tensor(response.embeddings).to('cuda')

    similarity = torch.matmul(query_embedding, doc_embeddings.T)
    most_similar_idx = torch.argmax(similarity)

    most_similar_doc = docs[most_similar_idx]
    most_similar_text = most_similar_doc['text']

    rag_outputs[query_id] = most_similar_text

with open('rag_outputs.json', 'w') as f:
    json.dump(rag_outputs, f)

print("RAG outputs saved to rag_outputs.json")
'''