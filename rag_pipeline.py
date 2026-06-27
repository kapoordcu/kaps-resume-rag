import requests
import numpy as np

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

EURI_API_KEY = os.getenv("EURI_API_KEY")

if not EURI_API_KEY:
    raise ValueError("EURI_API_KEY not found in .env")

def chunk_text(text, max_words=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))

    return chunks


model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def get_embedding(text):
    return model.encode(text, normalize_embeddings=True)



with open("candidate_story.txt", "r", encoding="utf-8") as f:
    raw_txt = f.read()
    chunks = chunk_text(raw_txt)
    print(f"Total Chunks: {len(chunks)}")
    # Example: Generate embedding for first chunk
    test_embedding = get_embedding(chunks[0])
    print(test_embedding)






