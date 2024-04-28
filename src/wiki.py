from datasets import load_dataset
import torch
import cohere
import os
import dotenv  
import dataloader

dotenv.load_dotenv()

co = cohere.Client(f"{os.getenv('COHERE_PROD')}")

docs_stream = load_dataset(f"Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True)

docs = []
doc_embeddings = []
loaded = 0

for doc in docs_stream:
    docs.append(doc)
    doc_embeddings.append(doc['emb'])
    loaded += 1
    print(f"Loaded {loaded} documents", end="\r")

doc_embeddings = torch.tensor(doc_embeddings)

rag_outputs = {}
queries = dataloader.get_questions()

for query_id, query in queries.items():
    response = co.embed(texts=[query], model='multilingual-22-12')
    query_embedding = torch.tensor(response.embeddings)

    similarity = torch.matmul(query_embedding, doc_embeddings.T)
    most_similar_idx = torch.argmax(similarity)

    most_similar_doc = docs[most_similar_idx]
    most_similar_text = most_similar_doc['text']

    rag_outputs[query_id] = most_similar_text

with open('rag_outputs.json', 'w') as f:
    json.dump(rag_outputs, f)

print("RAG outputs saved to rag_outputs.json")