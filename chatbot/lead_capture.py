"""
lead_capture.py
---------------
Handles all lead-related logic:
  - Dataclass definition for a Lead
  - Extracting structured lead data from raw conversation text using an LLM
  - Merging incremental updates into the active lead object
  - Delegating persistence to the database layer
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lead data model
# ---------------------------------------------------------------------------

@dataclass
class Lead:
    """Represents a single captured real estate lead."""

    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    budget: Optional[str] = None
    preferred_location: Optional[str] = None
    bhk_type: Optional[str] = None
    buying_timeline: Optional[str] = None
    notes: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(sep=" ", timespec="seconds"))

    # ------------------------------------------------------------------
    def is_complete(self) -> bool:
        """Return True if we have at least name + (phone or email)."""
        return bool(self.name) and bool(self.phone or self.email)

    # ------------------------------------------------------------------
    def missing_fields(self) -> list[str]:
        """Return a list of important fields that are still empty."""
        important = {
            "name": self.name,
            "phone": self.phone,
            "bhk_type": self.bhk_type,
            "budget": self.budget,
            "buying_timeline": self.buying_timeline,
        }
        return [k for k, v in important.items() if not v]

    # ------------------------------------------------------------------
    def merge(self, updates: dict) -> None:
        """Merge a dict of updates into this lead, ignoring null/None values."""
        for key, value in updates.items():
            if value and hasattr(self, key):
                setattr(self, key, value)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# LLM-based lead extraction
# ---------------------------------------------------------------------------

def extract_lead_from_conversation(conversation_text: str) -> dict:
    """
    Send the full conversation to the LLM and ask it to extract lead fields.
    Returns a dict that can be merged into a Lead object.
    Falls back to an empty dict on any error.
    """
    from chatbot.prompts import LEAD_EXTRACTION_PROMPT

    prompt = LEAD_EXTRACTION_PROMPT.format(conversation=conversation_text)

    try:
        if os.getenv("OPENAI_API_KEY"):
            result = _extract_with_openai(prompt)
        elif os.getenv("GEMINI_API_KEY"):
            result = _extract_with_gemini(prompt)
        else:
            logger.warning("No LLM API key set. Lead extraction skipped.")
            return {}

        # Parse JSON — the LLM is instructed to return ONLY JSON
        data = json.loads(result)
        # Replace JSON null with Python None
        return {k: (v if v != "null" else None) for k, v in data.items()}

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Lead extraction failed: {e}")
        return {}


def _extract_with_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


def _extract_with_gemini(prompt: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-flash-latest"))
    response = model.generate_content(prompt)
    return response.text.strip()


# ---------------------------------------------------------------------------
# Session-level lead manager
# ---------------------------------------------------------------------------

class LeadManager:
    """
    Manages the Lead object for a single chat session.
    Accumulates information incrementally as the conversation progresses.
    """

    def __init__(self):
        self.lead = Lead()
        self._saved = False

    # ------------------------------------------------------------------
    def update_from_conversation(self, conversation_history: list[dict]) -> None:
        """
        Re-extract lead data from the full conversation history and merge.
        Called after every assistant response.
        """
        # Build a plain-text transcript from the message list
        transcript = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation_history
        )
        updates = extract_lead_from_conversation(transcript)
        if updates:
            self.lead.merge(updates)
            logger.debug(f"Lead updated: {self.lead}")

    # ------------------------------------------------------------------
    def maybe_save(self, db) -> bool:
        """
        Save the lead to the database if it hasn't been saved yet and has
        at least minimal information (name + phone or email).
        Returns True if saved, False otherwise.
        """
        if not self._saved and self.lead.is_complete():
            db.save_lead(self.lead)
            self._saved = True
            logger.info(f"Lead saved: {self.lead.name}")
            return True
        return False

    # ------------------------------------------------------------------
    def force_save(self, db) -> None:
        """Save whatever we have, even if incomplete (end-of-session flush)."""
        if not self._saved and self.lead.name:
            db.save_lead(self.lead)
            self._saved = True
            logger.info(f"Lead force-saved (end of session): {self.lead.name}")

    # ------------------------------------------------------------------
    def get_summary(self) -> str:
        """Return a human-readable summary of captured lead info."""
        l = self.lead
        lines = ["📋 **Lead Summary**"]
        if l.name:
            lines.append(f"• Name: {l.name}")
        if l.phone:
            lines.append(f"• Phone: {l.phone}")
        if l.email:
            lines.append(f"• Email: {l.email}")
        if l.bhk_type:
            lines.append(f"• BHK: {l.bhk_type}")
        if l.budget:
            lines.append(f"• Budget: {l.budget}")
        if l.buying_timeline:
            lines.append(f"• Timeline: {l.buying_timeline}")
        if l.preferred_location:
            lines.append(f"• Location Pref: {l.preferred_location}")
        if l.notes:
            lines.append(f"• Notes: {l.notes}")
        return "\n".join(lines)
