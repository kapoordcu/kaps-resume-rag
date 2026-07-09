# kaps-resume-rag
pip install -r requirement.txt
source /Users/tea/gkaps/.venv/bin/activate

streamlit run gk_rag_app/app.py


# Docker Run
docker build -t gk-rag-app .
docker run --rm -p 8501:8501 --env-file /Users/tea/gkaps/gk_rag_app/.env gk-rag-app