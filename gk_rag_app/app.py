import streamlit as st
from utils.embedding import get_embedding
from utils.chunking import chunk_text
from utils.completion import generate_completion
from utils.prompt import build_prompt
from utils.retrival import load_faiss_index, retrieve_chunks


st.title("RAG App - GK story")
st.write("Ask about gaurav kapoor")

query = st.text_input("What you want to know about Gaurav")

if query:
    index, chunk_mapping = load_faiss_index()
    top_chunks = retrieve_chunks(query, index, chunk_mapping)
    prompt = build_prompt(top_chunks, query)
    response = generate_completion(prompt)

    st.subheader("answer")
    st.write(response)

    with st.expander("retrieved chunks"):
        for chunk in top_chunks:
            st.markdown(f" = {chunk}")