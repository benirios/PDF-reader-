import os
import json
import fitz  # PyMuPDF
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

# Carregar variáveis do .env (caso use)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "text-embedding-3-small"
FOLDER_PATH = "./pdfs"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

client = OpenAI(api_key=OPENAI_API_KEY)

def count_tokens(text, model="cl100k_base"):
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model=MODEL
    )
    return response.data[0].embedding

def process_all_pdfs(folder_path):
    all_results = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print(f"📄 Lendo {filename}")
            file_path = os.path.join(folder_path, filename)
            raw_text = extract_text_from_pdf(file_path)
            chunks = chunk_text(raw_text)
            print(f"✂️  {len(chunks)} chunks gerados")

            for i, chunk in enumerate(chunks):
                print(f"🧠 Gerando embedding do chunk {i+1}/{len(chunks)}")
                embedding = get_embedding(chunk)
                all_results.append({
                    "source": filename,
                    "chunk_index": i,
                    "text": chunk,
                    "embedding": embedding
                })

    return all_results

if __name__ == "__main__":
    results = process_all_pdfs(FOLDER_PATH)

    with open("embeddings.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Embeddings salvos em embeddings.json")
