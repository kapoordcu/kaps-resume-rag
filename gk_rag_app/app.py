import html

import streamlit as st

from utils.completion import generate_completion
from utils.prompt import build_prompt
from utils.retrival import load_faiss_index, retrieve_chunks


LINKEDIN_URL = "https://www.linkedin.com/in/kap6/"


st.set_page_config(
    page_title="Gaurav Kapoor | Resume RAG",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: #f7f8fb;
            color: #1f2937;
        }

        [data-testid="stHeader"] {
            background: rgba(247, 248, 251, 0.82);
        }

        .main .block-container {
            max-width: 900px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }

        .hero {
            background: linear-gradient(135deg, #ffffff 0%, #eef6f8 55%, #fff6ea 100%);
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 10px 28px rgba(17, 24, 39, 0.07);
        }

        .eyebrow {
            color: #0f766e;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
        }

        .hero h1 {
            color: #111827;
            font-size: clamp(1.6rem, 4vw, 2.6rem);
            line-height: 1;
            margin: 0 0 0.55rem;
        }

        .hero p {
            color: #4b5563;
            font-size: 0.95rem;
            line-height: 1.4;
            max-width: 720px;
            margin: 0;
        }

        .metric-row {
            display: grid;
            gap: 0.55rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin-top: 0.8rem;
        }

        .metric {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.6rem 0.7rem;
        }

        .metric strong {
            color: #111827;
            display: block;
            font-size: 0.98rem;
        }

        .metric span {
            color: #6b7280;
            font-size: 0.76rem;
        }

        .section-title {
            color: #111827;
            font-size: 1.1rem;
            font-weight: 750;
            margin: 0 0 0.35rem;
        }

        .helper {
            color: #6b7280;
            margin: 0 0 1rem;
        }

        .stTextInput > div > div > input {
            border-radius: 8px;
            border: 1px solid #d1d5db;
            padding: 0.85rem 1rem;
        }

        .stButton > button,
        .stLinkButton > a {
            border-radius: 8px;
            font-weight: 700;
            min-height: 2.75rem;
        }

        .answer-box {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-left: 5px solid #0f766e;
            border-radius: 8px;
            padding: 1.2rem 1.35rem;
            box-shadow: 0 10px 30px rgba(17, 24, 39, 0.06);
        }

        @media (max-width: 760px) {
            .hero {
                padding: 1rem;
            }

            .metric-row {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">Resume intelligence assistant</div>
        <h1>Curious about Gaurav Kapoor</h1>
        <p>
            Explore Gaurav's engineering background, cloud and microservices experience,
            leadership style, career goals, and personal story using a focused RAG assistant.
        </p>
        <div class="metric-row">
            <div class="metric"><strong>15+ years</strong><span>Software engineering experience</span></div>
            <div class="metric"><strong>Java / Springboot / Kafka / Cloud / Agentic Development</strong><span>Production system expertise</span></div>
            <div class="metric"><strong>Remote friendly</strong><span>Preferred next role</span></div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.write("")

left_col, right_col = st.columns([0.68, 0.32])

with left_col:
    st.markdown('<p class="section-title">What do you want to know?</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="helper">Ask a specific question and the app will retrieve the most relevant resume context before answering.</p>',
        unsafe_allow_html=True,
    )
    query = st.text_input(
        "Question",
        placeholder="Example: What kind of backend roles is Gaurav looking for?",
        label_visibility="collapsed",
    )
    submitted = st.button("Submit", type="primary", use_container_width=True)

with right_col:
    st.markdown('<p class="section-title">Profile</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="helper">Connect professionally or verify career details.</p>',
        unsafe_allow_html=True,
    )
    st.link_button("Open LinkedIn", LINKEDIN_URL, use_container_width=True)

if submitted and query.strip():
    st.write("")
    with st.status("Calculating response...", expanded=True) as status:
        st.write("Loading resume knowledge base")
        index, chunk_mapping = load_faiss_index()

        st.write("Finding the most relevant context")
        top_chunks = retrieve_chunks(query, index, chunk_mapping)

        st.write("Generating a grounded answer")
        prompt = build_prompt(top_chunks, query)
        response = generate_completion(prompt)

        status.update(label="Response ready", state="complete", expanded=False)

    st.markdown('<p class="section-title">Answer</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-box">{html.escape(response)}</div>', unsafe_allow_html=True)

    with st.expander("Retrieved context"):
        for chunk in top_chunks:
            st.markdown(f"- {chunk}")
