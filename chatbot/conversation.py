"""
conversation.py
---------------
Core conversation engine for the real estate chatbot.

Responsibilities:
  - Maintain per-session message history
  - Route user messages through the RAG pipeline
  - Call the LLM with system prompt + context + history
  - Trigger lead extraction after each exchange
  - Return the assistant's reply
"""

import logging
import os
from typing import List, Dict

from chatbot.prompts import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE, GREETING_MESSAGE
from chatbot.rag_pipeline import FAISSRetriever, retrieve_context
from chatbot.lead_capture import LeadManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

def _call_openai(system: str, messages: List[Dict]) -> str:
    """Call OpenAI ChatCompletion and return the assistant text."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    payload = [{"role": "system", "content": system}] + messages

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=payload,
        temperature=0.7,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


def _call_gemini(system: str, messages: List[Dict]) -> str:
    """Call Google Gemini and return the assistant text."""
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        system_instruction=system,
    )

    # Convert message history to Gemini format
    history = []
    for msg in messages[:-1]:   # all but the last (current) message
        history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]],
        })

    chat = model.start_chat(history=history)
    response = chat.send_message(messages[-1]["content"])
    return response.text.strip()


def _call_llm(system: str, messages: List[Dict]) -> str:
    """Route to the available LLM provider."""
    if os.getenv("OPENAI_API_KEY"):
        return _call_openai(system, messages)
    elif os.getenv("GEMINI_API_KEY"):
        return _call_gemini(system, messages)
    else:
        raise EnvironmentError(
            "No LLM API key found. Set OPENAI_API_KEY or GEMINI_API_KEY in .env"
        )


# ---------------------------------------------------------------------------
# Conversation session
# ---------------------------------------------------------------------------

class ConversationSession:
    """
    Encapsulates the full state of one user session:
      - message history (list of {role, content} dicts)
      - lead manager
      - retriever reference
    """

    def __init__(self, retriever: FAISSRetriever, db):
        self.retriever = retriever
        self.db = db
        self.history: List[Dict] = []
        self.lead_manager = LeadManager()
        self.turn_count: int = 0

    # ------------------------------------------------------------------
    def greeting(self) -> str:
        """Return the initial greeting without adding it to history yet."""
        return GREETING_MESSAGE

    # ------------------------------------------------------------------
    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the assistant reply.

        Steps:
          1. Append user message to history
          2. Retrieve relevant project context via RAG
          3. Build the prompt
          4. Call the LLM
          5. Append assistant reply to history
          6. Extract & update lead information
          7. Optionally save the lead to DB
        """
        self.turn_count += 1
        logger.info(f"Turn {self.turn_count} | User: {user_message[:80]}")

        # --- 1. Add user turn to history ---
        self.history.append({"role": "user", "content": user_message})

        # --- 2. Retrieve context ---
        try:
            context = retrieve_context(user_message, self.retriever)
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            context = ""

        # --- 3. Build history string for the prompt template ---
        history_text = "\n".join(
            f"{'Customer' if m['role'] == 'user' else 'Priya'}: {m['content']}"
            for m in self.history[:-1]   # exclude current user message (already in question)
        )

        # --- 4. Build messages for LLM ---
        # We inject RAG context into a specially formatted user message
        augmented_question = RAG_PROMPT_TEMPLATE.format(
            context=context if context else "No specific context retrieved.",
            history=history_text or "This is the start of the conversation.",
            question=user_message,
        )

        # For the LLM call we send: full history (minus last) + augmented last message
        llm_messages = self.history[:-1] + [{"role": "user", "content": augmented_question}]

        # --- 5. Call LLM ---
        try:
            reply = _call_llm(SYSTEM_PROMPT, llm_messages)
        except EnvironmentError as e:
            reply = (
                "I'm sorry, I'm having trouble connecting right now. "
                "Please call our sales team directly or visit the site. 🙏"
            )
            logger.error(str(e))
        except Exception as e:
            reply = "I'm sorry, something went wrong. Please try again in a moment."
            logger.error(f"LLM error: {e}")

        # --- 6. Append assistant reply ---
        self.history.append({"role": "assistant", "content": reply})

        # --- 7. Extract & save lead (every 2 turns to reduce API calls) ---
        if self.turn_count % 2 == 0:
            try:
                self.lead_manager.update_from_conversation(self.history)
                self.lead_manager.maybe_save(self.db)
            except Exception as e:
                logger.error(f"Lead capture error: {e}")

        logger.info(f"Turn {self.turn_count} | Assistant: {reply[:80]}")
        return reply

    # ------------------------------------------------------------------
    def end_session(self) -> str:
        """
        Called when the user ends the session.
        Force-saves lead and returns a farewell message with lead summary.
        """
        try:
            self.lead_manager.update_from_conversation(self.history)
            self.lead_manager.force_save(self.db)
        except Exception as e:
            logger.error(f"End-session lead save error: {e}")

        summary = self.lead_manager.get_summary()
        farewell = (
            "Thank you so much for your interest in **Seabreeze by Godrej Bayview**! 🏡\n\n"
            "Our team will reach out to you shortly. "
            "You're also welcome to visit our site any day between 10 AM – 7 PM.\n\n"
            f"{summary}\n\n"
            "Have a wonderful day! 😊"
        )
        return farewell
