from datasets import load_dataset
import torch
import cohere
import os
import dotenv  

dotenv.load_dotenv()

co = cohere.Client(f"{os.getenv('COHERE_PROD')}")

docs_stream = load_dataset(f"Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True)

docs = []
doc_embeddings = []
max_docs = 50000

for doc in docs_stream:
    docs.append(doc)
    doc_embeddings.append(doc['emb'])
    if len(docs) >= max_docs:
        break

doc_embeddings = torch.tensor(doc_embeddings)

queries = [
    "What is the capital of France?",
    "Who wrote the book '1984'?",
    "What is the most populous country in the world?",
    "What is the tallest mountain in the world?",
    "What is the longest river in the world?",
    "What is the smallest country in the world?",
    "What is the largest country in the world?",
    "What is the most spoken language in the world?",
    "What is the most popular sport in the world?",
    "What is the most popular movie of all time?",
    "What is the most popular song of all time?",
    "What is the most popular book of all time?",
    "What is the most popular video game of all time?",
    "What is the most popular TV show of all time?",
    "What is the most popular band of all time?",
    "What is the most popular artist of all time?",
    "What is the most popular food in the world?",
    "What is the most popular drink in the world?",
    "What is the most popular animal in the world?",
]

for query in queries:
    response = co.embed(texts=[query], model='multilingual-22-12')
    query_embedding = torch.tensor(response.embeddings)

    similarity = torch.matmul(query_embedding, doc_embeddings.T)
    most_similar_idx = torch.argmax(similarity)

    most_similar_doc = docs[most_similar_idx]
    most_similar_text = most_similar_doc['text']

    print(most_similar_text)