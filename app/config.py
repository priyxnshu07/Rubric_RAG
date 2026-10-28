"""Central settings, read from environment variables (and .env if present)."""
import os

from dotenv import load_dotenv

load_dotenv(override=True)  # the project's .env wins over stale keys set in the shell

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# "groq" or "gemini". If unset, use whichever provider has a key (Groq first).
LLM_PROVIDER = os.getenv("LLM_PROVIDER") or ("groq" if GROQ_API_KEY else "gemini")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"

# How many chunks to retrieve per grading request.
TOP_K_RUBRIC = int(os.getenv("TOP_K_RUBRIC", "8"))
TOP_K_NOTES = int(os.getenv("TOP_K_NOTES", "3"))
