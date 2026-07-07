import os
import requests
import numpy as np
from dotenv import load_dotenv
load_dotenv()

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