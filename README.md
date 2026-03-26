# Hospital Review RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot for hospital patient reviews. It uses CSV data, FAISS indexing, and (optionally) Google Gemini via LangChain.

## Features
- CSV loader + FAISS vector store
- Prompt-based summarization via language model
- Local fallback when API key or model access is unavailable
- Gradio UI (`--ui` command switch)

## Setup
1. Create and activate Python environment
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create env file(s):
- `.env` and/or `ke.env`

`ke.env` example:
```
GOOGLE_API_KEY=YOUR_ACTUAL_KEY
```

## Run
```bash
python Rag_Char.py "What are the main issues patients mention?"
```

UI mode:
```bash
python Rag_Char.py --ui
```

## Notes
- `faiss_index` directory will be generated on first run.
- `ke.env` overrides `.env`.
