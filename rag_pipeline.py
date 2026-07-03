import requests
import numpy as np
from google import genai
import os
import os
from dotenv import load_dotenv
import faiss

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

load_dotenv()


def chunk_text(text, max_words=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))

    return chunks


JINA_API_KEY = os.getenv("JINA_API_KEY")

def get_embedding(text):
    url = "https://api.jina.ai/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "jina-embeddings-v3",
        "input": [text]
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    return np.array(
        response.json()["data"][0]["embedding"],
        dtype=np.float32
    )

with open("candidate_story.txt", "r", encoding="utf-8") as f:
    raw_txt = f.read()

chunks = chunk_text(raw_txt)
print(f"Total Chunks: {len(chunks)}")

# Get one embedding to determine the dimension
test_embedding = get_embedding(chunks[0])
dimension = len(test_embedding)

index = faiss.IndexFlatL2(dimension)
chunk_mapping = []

for chunk in chunks:
    emb = get_embedding(chunk)  # Embed a single chunk

    # Convert to float32 and reshape to (1, dimension)
    emb = np.array(emb, dtype=np.float32).reshape(1, -1)

    index.add(emb)
    chunk_mapping.append(chunk)

print(f"Indexed {index.ntotal} chunks.")


def build_prompt(context_chunks, query):
    context = "\n\n".join(context_chunks)
    return f"""use the following context to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer:"""

def retrieve_top_k(query, k=4):
    query_embedding = get_embedding(query)
    query_embedding = np.array([query_embedding], dtype=np.float32)

    distances, indices = index.search(query_embedding, k)

    return [chunk_mapping[i] for i in indices[0] if i != -1]


client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_llm_response(prompt):

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text



query = "tell me who is gaurav"
top_chunks = retrieve_top_k(query=query)
prompt = build_prompt(top_chunks, query)
response = generate_llm_response(prompt)

print(response)