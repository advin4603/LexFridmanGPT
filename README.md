# Lex Fridman Podcast RAG based QnA System

An RAG-based Question Answering system over the lex fridman podcast dataset. 

# Requirements
- python3.10
- Qdrant Cloud Cluster
- Gemini API Key
- Other package Requirements in requirements.txt

# .env file contents
```
GOOGLE_API_KEY="<Google Api key>"
QDRANT_API_KEY="<Qdrant Api key>"
QDRANT_URL="<Qdrant cloud url>"
HAYHOOKS_ADDITIONAL_PYTHONPATH="."
```

# Instructions

- Run `podcast_rag.py` to generate pipeline dumps and generate document index.
- Run `export $(grep -v '^#' .env | xargs)` to set environment variables.
- Run `hayhooks run` to start server
- From a different terminal run `hayhooks deploy pipelines/rag_pipeline.yaml` to get the rag pipeline up and running
- Run `streamlit run chat_interface.py` to start the frontend server
- Visit `http://localhost:8501` for the QnA frontend.