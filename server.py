import os
import io
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from PyPDF2 import PdfReader

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configurações OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "text-embedding-3-small"
client = OpenAI(api_key=OPENAI_API_KEY)

# Google Drive config
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = "credentials.json"  # caminho para sua chave JSON

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

drive_service = build('drive', 'v3', credentials=credentials)

app = FastAPI()

# Exemplo de arquivo para baixar (ID do arquivo no Google Drive)
PDF_FILE_ID = "ID_DO_SEU_ARQUIVO_AQUI"

# Função para baixar PDF privado do Google Drive
def download_pdf(file_id, dest_path):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(dest_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")
    fh.close()

# Função para extrair texto do PDF
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

# Geração do embedding OpenAI
def get_embedding(text: str):
    response = client.embeddings.create(
        input=[text],
        model=MODEL
    )
    return response.data[0].embedding

# Similaridade cosseno
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Criar chunks e embeddings
def chunk_text(text, max_len=500):
    # Dividir texto em pedaços menores (exemplo simples por caracteres)
    chunks = []
    for i in range(0, len(text), max_len):
        chunks.append(text[i:i+max_len])
    return chunks

# Criar base de embeddings localmente
def build_embeddings(text):
    chunks = chunk_text(text)
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        embeddings.append({"chunk": chunk, "embedding": emb})
    return embeddings

# Buscar similaridade nos embeddings
def find_most_similar(query_embedding, embeddings_data, top_k=3):
    similarities = []
    for item in embeddings_data:
        sim = cosine_similarity(query_embedding, item["embedding"])
        similarities.append((sim, item))
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_results = similarities[:top_k]
    return [item for _, item in top_results]

# Carregar e indexar PDF na inicialização
PDF_LOCAL_PATH = "document.pdf"

if not os.path.exists(PDF_LOCAL_PATH):
    print("Baixando PDF do Google Drive...")
    download_pdf(PDF_FILE_ID, PDF_LOCAL_PATH)

print("Extraindo texto do PDF...")
text = extract_text_from_pdf(PDF_LOCAL_PATH)

print("Gerando embeddings do documento...")
embeddings_data = build_embeddings(text)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/query")
async def query(data: QueryRequest):
    try:
        query_embedding = get_embedding(data.question)
        results = find_most_similar(query_embedding, embeddings_data, top_k=data.top_k)
        response_chunks = [res["chunk"] for res in results]
        return {
            "question": data.question,
            "answers": response_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
