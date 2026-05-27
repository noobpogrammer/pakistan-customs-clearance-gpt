# Pakistan Customs AI Chatbot

A local RAG (Retrieval-Augmented Generation) chatbot built for Pakistan Customs. It lets you upload documents or crawl the customs website, then ask questions against that knowledge base using a locally-hosted LLM.

---

## Features

- Upload PDF, DOCX, and TXT documents and chat against their content
- Web spider that auto-crawls [customspk.com](http://customspk.com) and indexes all pages + PDFs
- Semantic search using FAISS vector index + `sentence-transformers`
- Multi-thread chat history stored in SQLite
- Runs fully offline using a local LLM (Meta-Llama-3 via local API)
- Clean browser-based chat UI served by Flask

---

## Project Structure

```
customgpt/
├── chatbot_server.py     # Flask server — document ingestion, vector search, LLM chat API + UI
├── scraper_ingest.py     # Spider that crawls customspk.com and feeds pages/PDFs into the index
├── requirements.txt      # Python dependencies
├── uploads/              # Uploaded/scraped documents (gitignored)
├── vector_index/         # FAISS index + SQLite DB (gitignored)
└── db/                   # Additional database files (gitignored)
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export API_URL="http://127.0.0.1:1337/v1/chat/completions"   # Your local LLM endpoint
export API_KEY="your_api_key_here"                            # API key if required
```

> The server connects to a local LLM that speaks the OpenAI chat completions API (e.g. LM Studio, Ollama, llama.cpp server). Default model: `Meta-Llama-3-8B-Instruct_IQ4_XS`.

### 3. Run the chatbot server

```bash
python chatbot_server.py
```

Then open `http://localhost:5000` in your browser.

### 4. (Optional) Crawl the customs website

```bash
python scraper_ingest.py
```

This will spider up to 50 pages on customspk.com, extract text, download PDFs, and add everything to the knowledge base automatically.

---

## How It Works

1. **Document Ingestion** — uploaded files are chunked (1200 chars, 300 overlap) and embedded with `all-MiniLM-L6-v2`
2. **Vector Store** — embeddings are stored in a FAISS index; chunk text and metadata go into SQLite
3. **Query** — user questions are embedded and the top-k most relevant chunks are retrieved
4. **Generation** — retrieved context + conversation history are sent to the local LLM to generate an answer

---

## Tech Stack

| Component | Library |
|---|---|
| Web framework | Flask |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS |
| PDF parsing | pymupdf4llm |
| DOCX parsing | python-docx |
| Web scraping | BeautifulSoup4, requests |
| Database | SQLite3 |
| LLM | Any local OpenAI-compatible API |

---

## Author

**Mohammad Mukadam**
