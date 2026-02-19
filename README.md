# 📘 ResearchRAG-AI — Advanced AI Research Assistant

> An intelligent, document-grounded RAG (Retrieval-Augmented Generation) system that answers questions **only** from the documents you provide — with **zero hallucination**.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.1-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔗 **Multi-Source Ingestion** | Load from URLs, PDFs, DOCX, and TXT files simultaneously |
| 🔀 **Hybrid Search** | BM25 (keyword) + FAISS (semantic) via `EnsembleRetriever` |
| 🧠 **6-Layer Anti-Hallucination** | Pipeline to detect and block fabricated answers |
| 🤖 **LLM Intent Classifier** | AI-powered check to reject nonsensical/off-topic questions |
| 📚 **Source Citations** | Every answer cites the exact document/URL it came from |
| ☁️ **Cloud + Local LLM** | Switch between Groq API (cloud) and Ollama (local) |

---

## 🛡️ Anti-Hallucination Pipeline (6 Layers)

```
User Query
    ↓
Layer 1    → Regex Guard         (blocks greetings, slurs, identity questions)
    ↓
Layer 1.5  → LLM Intent Check    (RESEARCH / MIXED / NONSENSE classifier)
    ↓
Layer 2    → FAISS Relevance Gate (rejects if no document chunk is close enough)
    ↓
Layer 3    → Hybrid Retrieval    (60% FAISS semantic + 40% BM25 keyword)
    ↓
Layer 4    → Strict Prompt       (LLM forbidden from using external knowledge)
    ↓
Layer 5    → Post-Processing     (catches any "no info" leakage in LLM response)
    ↓
    ✅ Grounded Answer with Citations
```

---

## 🤖 LLM Providers

### ☁️ Cloud — Groq API (Llama 3.1 8B Instant)
- **Get your free API key:** [https://console.groq.com/home](https://console.groq.com/home)
- Fast inference, 14,400 requests/day free tier
- Model: `llama-3.1-8b-instant`

### 🖥️ Local — Ollama (Llama 3.1)
- **Install Ollama:** [https://ollama.ai](https://ollama.ai)
- Pull the model: `ollama pull llama3.1`
- Runs 100% offline, no API key needed

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/V2A0R0U4N/ResearchRAG-AI.git
cd ResearchRAG-AI
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
OLLAMA_BASE_URL=http://localhost:11434    # optional, only for local Ollama
```
Get your Groq API key for free at: [https://console.groq.com/home](https://console.groq.com/home)

### 5. Run the app
```bash
streamlit run main.py
```

---

## 📦 Requirements

```
streamlit
python-dotenv
langchain>=0.1.17
langchain-community>=0.0.25
langchain-openai>=0.0.6
langchain-huggingface>=0.0.3
langchain-groq>=0.1.0
langchain-ollama>=0.1.0
langchain-groq>=0.1.0
faiss-cpu
sentence-transformers
unstructured
pypdf>=3.0.0
docx2txt>=0.8
rank_bm25>=0.2.2
requests
tqdm
beautifulsoup4
```

---

## 📁 Project Structure

```
ResearchRAG-AI/
├── main.py              # Streamlit app — full RAG pipeline
├── query_guard.py       # Layer 1 regex guard + Layer 2 relevance gate
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not committed to git)
├── .gitignore           # Excludes venv, .env, .pkl files
└── README.md
```

---

## 🧪 How to Use

1. **Add sources** — Paste article URLs (one per field) or upload PDF/DOCX/TXT files
2. **Click "Process Research Material"** — Documents are chunked, embedded, and indexed
3. **Ask a question** — Type a specific research question about your documents
4. **Get a grounded answer** — The system retrieves relevant chunks and generates a cited response

> **Note:** The system will politely decline if your question is off-topic, nonsensical, or not covered in the provided documents.

---

## 🔧 Configuration

| Parameter | Default | Description |
|---|---|---|
| LLM Temperature | `0.1` | Low for factual accuracy |
| Chunk Size | `600` chars | Document splitting size |
| Chunk Overlap | `150` chars | Context continuity |
| Retrieval k | `8` | Chunks fetched per query |
| FAISS Threshold | `1.8` | L2 distance cutoff for relevance |
| Hybrid Weights | `60/40` | FAISS semantic / BM25 keyword |

---

## 📄 License

MIT License — free to use, modify, and distribute.
