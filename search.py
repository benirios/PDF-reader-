import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "text-embedding-3-small"

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model=MODEL
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_most_similar(query_embedding, embeddings_data, top_k=3):
    similarities = []

    for item in embeddings_data:
        sim = cosine_similarity(query_embedding, item["embedding"])
        similarities.append((sim, item))

    similarities.sort(key=lambda x: x[0], reverse=True)
    top_results = similarities[:top_k]
    return [item for _, item in top_results]

if __name__ == "__main__":
    # Carregar os embeddings já gerados
    with open("embeddings.json", "r") as f:
        embeddings_data = json.load(f)

    query = input("Digite sua pergunta: ")
    query_embedding = get_embedding(query)

    results = find_most_similar(query_embedding, embeddings_data, top_k=3)

    print("\nResultados mais relevantes:\n")
    for r in results:
        print(f"Fonte: {r['source']} | Chunk: {r['chunk_index']}")
        print(r['text'])
        print("------")
