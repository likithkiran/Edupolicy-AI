# 🎓 EduPolicy AI

A **Retrieval-Augmented Generation (RAG)** application for semantic search and AI-powered Q&A over academic policy PDF documents.

---

## Features

| Capability | Technology |
|---|---|
| PDF text extraction | **PyMuPDF** (fitz) |
| Semantic embeddings | **Sentence Transformers** (`all-MiniLM-L6-v2`) |
| Similarity search | **NumPy** cosine similarity |
| Answer generation | **Google Gemini API** |
| Web UI | **Streamlit** |

- 📄 Upload multiple PDFs directly in the browser
- 🔍 Per-page semantic indexing with original page number preservation
- 📌 Every answer is cited with **document name + page number**
- 💬 Persistent in-session chat history
- ⚡ Runs fully on CPU — no GPU required

---

## Project Structure

```
EduPolicy-AI/
├── app.py              # Streamlit UI and session orchestration
├── pdf_processor.py    # PDF extraction → PageChunk objects
├── search_engine.py    # Sentence Transformer embedding + cosine search
├── rag_engine.py       # Gemini API integration + prompt construction
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a Gemini API key

Visit [Google AI Studio](https://aistudio.google.com/app/apikey) and create a free API key.

### 3. Launch the app

```bash
streamlit run app.py
```

### 4. Use the app

1. Paste your **Gemini API key** in the sidebar.
2. **Upload** one or more academic policy PDFs.
3. Click **Index Documents** — the app extracts and embeds all pages.
4. Type a **question** and click **Ask**.
5. Read the AI-generated answer with **cited sources**.

---

## How It Works

```
PDF Upload
    │
    ▼
pdf_processor.py  ──►  PageChunk(doc_name, page_number, text)
    │
    ▼
search_engine.py  ──►  SentenceTransformer embeddings (all-MiniLM-L6-v2)
                        NumPy cosine similarity index
    │
    ▼  (query time)
User Question  ──►  Top-K relevant pages retrieved
    │
    ▼
rag_engine.py  ──►  Context block + Gemini prompt
    │
    ▼
Cited Answer  ──►  Streamlit UI
```

---

## Citation Format

Every passage and answer is cited as:

```
[Document Name, p. 12]
```

The document name is the PDF filename (without `.pdf`), and the page number is the **original page number** from the source document.

---

## Configuration

| Sidebar Option | Description |
|---|---|
| Gemini API Key | Your Google AI Studio key |
| Passages to retrieve | Top-K chunks fed to Gemini (1–10) |
| Gemini model | `gemini-1.5-flash` (fast) or `gemini-1.5-pro` (accurate) |

---

## Requirements

- Python 3.10+
- Internet connection (for Gemini API calls)
- No GPU required

---

## License

MIT License — free to use and modify for academic and research purposes.
