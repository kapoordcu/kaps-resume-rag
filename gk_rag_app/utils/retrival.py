import os
import faiss
import pickle
import numpy as np
from chunking import chunk_text
from embedding import get_embedding

def load_faiss_index():
    if os.path.exists("faiss_store/index.faiss"):
        index = faiss.read_index("faiss_store/index.faiss")
        with open("faiss_store/chunk_mapping.pkl", "rb") as f:
            chunk_mapping = pickle.load(f)
    else:
        with open("../data/candidate_story.txt", "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        chunk_mapping = []
        index = faiss.IndexFlatL2(251)
        for chunk in chunks:
            emb = get_embedding(chunk)  # Embed a single chunk
            # Convert to float32 and reshape to (1, dimension)
            emb = np.array(emb, dtype=np.float32).reshape(1, -1)
            index.add(emb)
            chunk_mapping.append(chunk)
        os.makedirs("faiss_store", exist_ok=True)
        faiss.write_index(index, "faiss_store/index.faiss")
        with open("faiss_store/chunk_mapping.pkl", "rb") as f:
            pickle.dump(chunk_mapping, f)
    return index, chunk_mapping

def retrieve_chunks(query, index, chunk_mapping):
    query_embedding = get_embedding(query)
    query_embedding = np.array([query_embedding], dtype=np.float32)

    distances, indices = index.search(query_embedding, k)

    return [chunk_mapping[i] for i in indices[0] if i != -1]