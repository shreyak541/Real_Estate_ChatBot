# 🏙️ Seabreeze Real Estate Chatbot

**AI-powered lead generation chatbot for Seabreeze by Godrej Bayview, Vashi.**

Built with FastAPI · Streamlit · OpenAI / Gemini · FAISS RAG · SQLite / CSV

---

## ✨ Features

| Feature | Details |
|---|---|
| 💬 Conversational AI | GPT-4o-mini or Gemini 1.5 Flash powered chat |
| 🔍 RAG Pipeline | FAISS vector search over project knowledge base |
| 📋 Lead Capture | Automatically extracts Name, Phone, Budget, BHK, Timeline |
| 💾 Lead Storage | CSV (default) or SQLite — switchable via .env |
| 🖥️ Web UI | Beautiful Streamlit chat interface |
| 🚀 REST API | FastAPI backend with session management |
| 📥 Export | Download captured leads as CSV |

---

## 📁 Project Structure

```
real_estate_chatbot/
│
├── app.py                   # FastAPI REST API
├── streamlit_app.py         # Streamlit chat UI (recommended entry point)
│
├── chatbot/
│   ├── __init__.py
│   ├── conversation.py      # Core conversation engine (RAG + LLM + history)
│   ├── lead_capture.py      # Lead dataclass, extraction, manager
│   ├── rag_pipeline.py      # FAISS index builder and retriever
│   └── prompts.py           # All LLM prompt templates
│
├── data/
│   ├── project_details.txt  # Knowledge base (edit to update project info)
│   ├── leads.csv            # Captured leads (auto-created)
│   └── faiss_index.pkl      # Cached FAISS index (auto-created)
│
├── database/
│   ├── __init__.py
│   └── db.py                # CSV and SQLite backends
│
├── utils/
│   ├── __init__.py
│   └── helpers.py           # Logging, env loading, utility functions
│
├── requirements.txt
├── .env.example             # Copy to .env and fill in your keys
└── README.md
```

---

## ⚙️ Prerequisites

- Python **3.11+**
- An **OpenAI API key** (GPT-4o-mini) **OR** a **Google Gemini API key**
- ~500 MB disk space (for FAISS and dependencies)

---

## 🚀 Setup & Run

### 1. Clone / extract the project

```bash
cd real_estate_chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Open .env and add your OpenAI or Gemini API key
```

Minimum required in `.env`:
```
OPENAI_API_KEY=sk-...       # OR
GEMINI_API_KEY=...
```

### 5a. Run the Streamlit UI (recommended)

```bash
streamlit run streamlit_app.py
```
Open http://localhost:8501 in your browser.

### 5b. Run the FastAPI backend (optional, for API-only usage)

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
API docs available at http://localhost:8000/docs

---

## 🤖 How It Works

```
User message
    ↓
FAISS retriever searches project_details.txt
    ↓
Top-4 relevant chunks selected as context
    ↓
Context + conversation history + user message → LLM
    ↓
LLM generates Priya's reply
    ↓
Lead extraction (every 2 turns) → stored to CSV/SQLite
    ↓
Reply returned to user
```

### RAG Pipeline Detail

1. `project_details.txt` is chunked into 300-character overlapping segments.
2. Each chunk is embedded using OpenAI `text-embedding-3-small` (or TF-IDF fallback).
3. Embeddings are stored in a FAISS flat index, cached to `data/faiss_index.pkl`.
4. On each query, the question is embedded and cosine similarity retrieves the top 4 chunks.
5. These chunks are injected into the LLM prompt as project context.

### Lead Capture Detail

1. After every 2 turns, the full conversation transcript is sent to the LLM.
2. The LLM extracts structured JSON: `{name, phone, email, budget, bhk_type, ...}`.
3. Extracted values are merged into the session's `Lead` object.
4. Once `name` + `phone/email` are captured, the lead is saved automatically.
5. At session end (`End` button), any remaining data is force-saved.

---

## 🛠️ Configuration Options

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI key (use this OR Gemini) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `GEMINI_API_KEY` | — | Gemini key (use this OR OpenAI) |
| `GEMINI_MODEL` | `gemini-flash-latest` | Gemini model to use |
| `DB_BACKEND` | `csv` | `csv` or `sqlite` |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## 📊 Viewing Leads

**Via Streamlit:** Click "End" → "⬇️ Download Leads" button.

**Via FastAPI:**
```
GET http://localhost:8000/leads
GET http://localhost:8000/leads/export   # downloads CSV
```

**Via Python:**
```python
from database.db import LeadDatabase
db = LeadDatabase()
df = db.get_all_leads()
print(df)
```

---

## 🔧 Customisation

### Update project information
Edit `data/project_details.txt` and delete `data/faiss_index.pkl` to force a re-index.

### Add a new LLM provider
Add a `_call_<provider>()` function in `chatbot/conversation.py` and update `_call_llm()`.

### Change the chatbot persona
Edit the `SYSTEM_PROMPT` in `chatbot/prompts.py`.

### Switch to SQLite
Set `DB_BACKEND=sqlite` in `.env`.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `No LLM API key found` | Set `OPENAI_API_KEY` or `GEMINI_API_KEY` in `.env` |
| FAISS import error | Run `pip install faiss-cpu` |
| Slow first start | Normal — FAISS index is being built. Cached after first run. |
| Lead not saved | Ensure name + phone/email are shared in chat. Click "End" to force-save. |
| TF-IDF fallback | Add an OpenAI key for proper semantic embeddings in production. |

---

## 📝 Sample Conversation

```
Priya: Namaste! Welcome to Seabreeze by Godrej Bayview, Vashi. I'm Priya...
       May I start by knowing your name? 😊

User:  Hi, I'm Rahul

Priya: Lovely to meet you, Rahul! Are you looking at a 2 BHK or 3 BHK?

User:  3 BHK. What's the price?

Priya: Great choice! Our 3 BHK residences start at ₹4.75 Cr for 1266+ sq ft
       with a private deck and stunning views. May I ask your budget range?

User:  Around 5 Cr. My number is 9876543210.

Priya: Perfect, Rahul. I've noted ₹5 Cr budget and will have our team call you
       at 9876543210. Would you like to schedule a site visit this weekend?
```

---

## 📄 License

This project is a prototype for demonstration purposes.
Project details belong to Godrej Properties Ltd.

---

*Built with ❤️ for the Indian real estate market.*
