"""Central settings, read from environment variables (and .env if present)."""
import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"

# How many chunks to retrieve per grading request.
TOP_K_RUBRIC = int(os.getenv("TOP_K_RUBRIC", "8"))
TOP_K_NOTES = int(os.getenv("TOP_K_NOTES", "3"))
