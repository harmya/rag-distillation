from datasets import load_dataset
import torch
import cohere
import os
import dotenv  

dotenv.load_dotenv()

co = cohere.Client(f"{os.getenv('COHERE_API_KEY')}")

docs_stream = load_dataset(f"Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True)

docs = []
doc_embeddings = []

for doc in docs_stream:
    docs.append(doc)
    doc_embeddings.append(doc['emb'])

doc_embeddings = torch.tensor(doc_embeddings)

query = "What is the capital of France?"
response = co.embed(texts=[query], model='multilingual-22-12')
query_embedding = torch.tensor(response.embeddings)

similarity = torch.matmul(query_embedding, doc_embeddings.T)
most_similar_idx = torch.argmax(similarity)

most_similar_doc = docs[most_similar_idx]
most_similar_text = most_similar_doc['text']

print(most_similar_text)
