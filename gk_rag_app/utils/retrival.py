import os
import faiss
import pickle
import numpy as np
from utils.chunking import chunk_text
from utils.embedding import get_embedding


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_STORE_DIR = os.path.join(BASE_DIR, "faiss_store")
INDEX_PATH = os.path.join(FAISS_STORE_DIR, "index.faiss")
CHUNK_MAPPING_PATH = os.path.join(FAISS_STORE_DIR, "chunk_mapping.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "candidate_story.txt")


def load_faiss_index():
    if _has_saved_index():
        try:
            index = faiss.read_index(INDEX_PATH)
            with open(CHUNK_MAPPING_PATH, "rb") as f:
                chunk_mapping = pickle.load(f)
            return index, chunk_mapping
        except (RuntimeError, EOFError, pickle.UnpicklingError):
            pass

    return build_faiss_index()


def _has_saved_index():
    return (
        os.path.exists(INDEX_PATH)
        and os.path.getsize(INDEX_PATH) > 0
        and os.path.exists(CHUNK_MAPPING_PATH)
        and os.path.getsize(CHUNK_MAPPING_PATH) > 0
    )


def build_faiss_index():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    chunk_mapping = []
    index = None

    for chunk in chunks:
        emb = get_embedding(chunk)  # Embed a single chunk
        emb = np.array(emb, dtype=np.float32).reshape(1, -1)
        if index is None:
            index = faiss.IndexFlatL2(emb.shape[1])
        index.add(emb)
        chunk_mapping.append(chunk)

    if index is None:
        raise ValueError(f"No chunks were created from {DATA_PATH}")

    os.makedirs(FAISS_STORE_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNK_MAPPING_PATH, "wb") as f:
        pickle.dump(chunk_mapping, f)

    return index, chunk_mapping

def retrieve_chunks(query, index, chunk_mapping, k=3):
    query_embedding = get_embedding(query)
    query_embedding = np.array([query_embedding], dtype=np.float32)

    _, indices = index.search(query_embedding, min(k, len(chunk_mapping)))

    return [chunk_mapping[i] for i in indices[0] if i != -1]
