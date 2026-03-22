"""
app.py
------
FastAPI application entry point for the Seabreeze Real Estate Chatbot.

Endpoints:
  GET  /          → Health check
  POST /chat      → Send a message and get a reply
  POST /end       → End a session and save lead
  GET  /leads     → View all captured leads (admin)
  GET  /leads/export → Download leads as CSV

Sessions are stored in-memory (keyed by session_id).
For production, replace with Redis or a persistent session store.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Project imports
from utils.helpers import configure_logging, load_env
from chatbot.rag_pipeline import get_retriever
from chatbot.conversation import ConversationSession
from database.db import LeadDatabase

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

load_env()
configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Build the FAISS retriever once at startup (shared across all sessions)
logger.info("Initialising RAG retriever …")
retriever = get_retriever()

# Single shared DB instance
db = LeadDatabase()

# In-memory session store: session_id → ConversationSession
sessions: Dict[str, ConversationSession] = {}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Seabreeze Real Estate Chatbot API",
    description="AI-powered lead generation chatbot for Seabreeze by Godrej Bayview, Vashi.",
    version="1.0.0",
)

# Allow the Streamlit frontend (or any origin during development) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str | None = None   # If None, a new session is created
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    lead_complete: bool             # True once name + contact are captured


class EndSessionResponse(BaseModel):
    session_id: str
    farewell: str
    lead_saved: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_create_session(session_id: str | None) -> tuple[str, ConversationSession]:
    """Return an existing session or create a new one."""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]

    new_id = session_id or str(uuid.uuid4())
    session = ConversationSession(retriever=retriever, db=db)
    sessions[new_id] = session
    logger.info(f"New session created: {new_id}")
    return new_id, session


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["ui"])
def serve_ui():
    """Serve the chat UI — open http://localhost:8000 in your browser."""
    html_path = Path(__file__).resolve().parent / "chat.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>chat.html not found. Place it next to app.py.</h2>", status_code=404)


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok", "project": "Seabreeze Real Estate Chatbot"}


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest):
    """
    Accept a user message, run RAG + LLM, return the assistant reply.
    If no session_id is provided, a new session is started automatically.
    The first message triggers a greeting prepended to the reply.
    """
    session_id, session = _get_or_create_session(request.session_id)

    # On the very first turn, prepend the greeting
    if session.turn_count == 0:
        greeting = session.greeting()
        # Add greeting to history as an assistant message so the LLM has context
        session.history.append({"role": "assistant", "content": greeting})
        # Process the actual first user message
        reply = session.chat(request.message)
        # Return greeting + reply as one combined message only if the user
        # sent something (not an empty "start" trigger)
        if request.message.strip():
            full_reply = reply
        else:
            full_reply = greeting
    else:
        reply = session.chat(request.message)
        full_reply = reply

    return ChatResponse(
        session_id=session_id,
        reply=full_reply,
        lead_complete=session.lead_manager.lead.is_complete(),
    )


@app.post("/start", tags=["chat"])
def start_session():
    """Create a new session and return the greeting message."""
    session_id, session = _get_or_create_session(None)
    greeting = session.greeting()
    session.history.append({"role": "assistant", "content": greeting})
    return {"session_id": session_id, "reply": greeting}


@app.post("/end", response_model=EndSessionResponse, tags=["chat"])
def end_session(session_id: str):
    """End the session, save the lead, and return a farewell message."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[session_id]
    farewell = session.end_session()
    lead_saved = session.lead_manager._saved

    # Clean up session from memory
    del sessions[session_id]
    logger.info(f"Session ended: {session_id}")

    return EndSessionResponse(
        session_id=session_id,
        farewell=farewell,
        lead_saved=lead_saved,
    )


@app.get("/leads", tags=["admin"])
def get_leads():
    """Return all captured leads as a list of dicts."""
    df = db.get_all_leads()
    return {"count": len(df), "leads": df.to_dict(orient="records")}


@app.get("/leads/export", tags=["admin"])
def export_leads():
    """Download all leads as a CSV file."""
    csv_path = Path(__file__).resolve().parent / "data" / "leads.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="No leads file found.")
    return FileResponse(
        path=str(csv_path),
        media_type="text/csv",
        filename="seabreeze_leads.csv",
    )
