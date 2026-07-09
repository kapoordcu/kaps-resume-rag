# kaps-resume-rag
pip install -r requirement.txt
source /Users/tea/gkaps/.venv/bin/activate

streamlit run gk_rag_app/app.py


# Docker Run
docker build -t gk-rag-app .
docker run --rm -p 8501:8501 -e GOOGLE_API_KEY="your_key_here" gk-rag-app