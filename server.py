import json
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "text-embedding-3-small"
MAX_CHAR_LENGTH = 400  # limite de caracteres para a resposta

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

# Carregar embeddings
with open("embeddings.json", "r") as f:
    embeddings_data = json.load(f)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 1

def get_embedding(text: str):
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

def shorten_text(text, max_length=MAX_CHAR_LENGTH):
    return text if len(text) <= max_length else text[:max_length] + "..."

@app.post("/query")
async def query(data: QueryRequest):
    try:
        query_embedding = get_embedding(data.question)
        results = find_most_similar(query_embedding, embeddings_data, top_k=data.top_k)
        best_answer = results[0]["text"] if results else "Desculpe, não encontrei nada relevante."
        return {
            "question": data.question,
            "answer": shorten_text(best_answer)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
