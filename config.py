# ─────────────────────────────────────────────────────────────────────────────
# config.py  –  Central Configuration for AI Interview Preparation Bot
#
# PURPOSE:
#   All project-wide settings live here so that changing one value
#   automatically updates every module that imports it.
#   Uses python-dotenv to load secrets from a .env file, keeping
#   API keys out of source control.
# ─────────────────────────────────────────────────────────────────────────────

import os                       # Standard library – access environment variables & paths
from dotenv import load_dotenv  # python-dotenv: reads key=value pairs from .env file

# ── Load .env file ────────────────────────────────────────────────────────────
# load_dotenv() searches for a file called .env in the current directory
# (or parent directories) and injects its contents into os.environ.
# After this call, os.getenv("KEY") returns whatever is in your .env file.
load_dotenv()


# ═════════════════════════════════════════════════════════════════════════════
# 1.  API KEYS  (read from .env – never hard-code secrets in source code!)
# ═════════════════════════════════════════════════════════════════════════════
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
# ^ os.getenv(key, default) – returns the value or "" if the key is missing.
#   The type annotation ": str" is just documentation; Python doesn't enforce it.


# ═════════════════════════════════════════════════════════════════════════════
# 2.  GEMINI MODEL SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
GEMINI_MODEL: str = "gemini-3.8-flash"
# The model ID used for all Gemini API calls.
# "gemini-2.0-flash" is fast and cost-efficient – good for real-time chat.

GEMINI_TEMPERATURE: float = 0.7
# Controls randomness:  0.0 = deterministic / robotic,  1.0 = very creative.
# 0.7 gives a balanced, natural-sounding response.

GEMINI_MAX_TOKENS: int = 1024
# Maximum number of tokens (≈ words) the model will generate per response.


# ═════════════════════════════════════════════════════════════════════════════
# 3.  INTERVIEW DOMAINS  (shown as tabs / dropdown options in the UI)
# ═════════════════════════════════════════════════════════════════════════════
INTERVIEW_DOMAINS: list[str] = [
    "Software Engineering",
    "Data Science & ML",
    "System Design",
    "Behavioral / HR",
    "Product Management",
    "DevOps & Cloud",
    "Cybersecurity",
    "Frontend Development",
    "Backend Development",
    "Full Stack Development",
]
# list[str] is a generic type hint (Python 3.9+).  It says "a list whose
# items are all strings".  Used purely for readability / IDE support.


# ═════════════════════════════════════════════════════════════════════════════
# 4.  DIFFICULTY LEVELS
# ═════════════════════════════════════════════════════════════════════════════
DIFFICULTY_LEVELS: list[str] = ["Beginner", "Intermediate", "Advanced"]


# ═════════════════════════════════════════════════════════════════════════════
# 5.  AUDIO SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
SAMPLE_RATE: int = 16_000       # Audio sample rate in Hz (16 kHz is standard for speech)
AUDIO_CHANNELS: int = 1         # Mono audio (1 channel is enough for voice)
CHUNK_SIZE: int = 1024          # How many audio frames are read per buffer iteration
SILENCE_THRESHOLD: float = 0.01 # RMS amplitude below which we consider it "silence"
SILENCE_DURATION: float = 2.0   # Seconds of silence before auto-stopping recording

TEMP_AUDIO_DIR: str = "temp_audio"   # Folder for storing temporary .wav files
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
# os.makedirs with exist_ok=True: creates the folder if it doesn't exist,
# and silently does nothing if it already exists (no error).


# ═════════════════════════════════════════════════════════════════════════════
# 6.  NLP / SCORING SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
MIN_ANSWER_WORDS: int = 10      # Answers shorter than this are flagged as "too brief"
FILLER_WORDS: list[str] = [     # Common speech disfluencies to detect & penalise
    "um", "uh", "like", "you know", "basically", "literally",
    "actually", "so", "right", "okay", "well",
]

# TF-IDF similarity threshold – if cosine similarity between answer and
# ideal answer is above this value, the answer is considered "on-topic".
SIMILARITY_THRESHOLD: float = 0.3


# ═════════════════════════════════════════════════════════════════════════════
# 7.  STREAMLIT PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
PAGE_TITLE: str = "AI Interview Preparation Bot"
PAGE_ICON: str = "🤖"
LAYOUT: str = "wide"            # "centered" | "wide"
