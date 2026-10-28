# RubricRAG: API (port 8000) + Streamlit UI (port 8501) in one container.
#   docker build -t rubricrag .
#   docker run --env-file .env -p 8501:8501 -p 8000:8000 rubricrag
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# CPU-only torch first, so sentence-transformers doesn't pull the much larger CUDA build.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

# Bake the embedding model into the image so the container starts without downloading it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

COPY app/ app/
COPY data/ data/

# Load the sample rubrics into ChromaDB at build time.
RUN python -m app.ingest data/biology-quiz-1 && python -m app.ingest data/public_speaking

EXPOSE 8000 8501
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & \
     API_URL=http://localhost:8000 streamlit run app/ui.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]
