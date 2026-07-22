"""Environment-driven configuration for the unified Merchandise AI server.

Covers the voice sidecar knobs (STT / TTS / VAD / audit store / CORS) plus the
LLM layer (provider-switchable, two roles: vision and chat). All paths default
relative to the ``python-server/`` directory (not the process CWD), and ``.env`` is
loaded from ``python-server/.env``.
"""
from __future__ import annotations

import os
from pathlib import Path

# python-server/app/config.py -> APP_DIR = python-server/app, SERVER_DIR = python-server
APP_DIR = Path(__file__).resolve().parent
SERVER_DIR = APP_DIR.parent

try:
    from dotenv import load_dotenv

    # Load python-server/.env explicitly so config is stable regardless of CWD.
    load_dotenv(SERVER_DIR / ".env")
except Exception:  # python-dotenv is optional at runtime
    pass


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_path(env_value: str | None, *default_parts: str) -> str:
    """Resolve a filesystem path relative to ``SERVER_DIR`` when it is relative.

    An absolute env value is honored as-is; a relative env value is anchored to
    ``python-server/`` (not the CWD); an unset value uses the given default under
    ``python-server/``.
    """
    if env_value:
        p = Path(env_value)
        if not p.is_absolute():
            p = SERVER_DIR / p
        return str(p)
    return str(SERVER_DIR.joinpath(*default_parts))


# --------------------------------------------------------------------------- #
# Server.
# --------------------------------------------------------------------------- #
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8123"))

# --------------------------------------------------------------------------- #
# Speech-to-text (faster-whisper).
# --------------------------------------------------------------------------- #
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en") or None

# The browser captures and downsamples mic audio to this rate before streaming.
STT_SAMPLE_RATE = 16000

# Live interim transcripts: periodically re-decode the in-progress buffer with a
# fast (greedy) pass and stream `partial` events to the client.
PARTIAL_ENABLED = os.getenv("PARTIAL_ENABLED", "true").lower() in ("1", "true", "yes")
PARTIAL_INTERVAL_MS = float(os.getenv("PARTIAL_INTERVAL_MS", "900"))

# --------------------------------------------------------------------------- #
# Text-to-speech (Kyutai Pocket TTS — replaces Kokoro).
# --------------------------------------------------------------------------- #
POCKET_TTS_VOICE = os.getenv("POCKET_TTS_VOICE", "alba")

# --------------------------------------------------------------------------- #
# Open-mic / hands-free turn detection.
# Silero VAD — frame-level speech detection.
# --------------------------------------------------------------------------- #
SILERO_MODEL_PATH = _resolve_path(
    os.getenv("SILERO_MODEL_PATH"), "models", "silero_vad.onnx"
)
VAD_FRAME_SAMPLES = 512  # 32 ms @ 16 kHz (Silero's required frame size)
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
# Trailing silence before a turn can end (without Smart Turn) / hard cap (with it).
VAD_MIN_SILENCE_MS = float(os.getenv("VAD_MIN_SILENCE_MS", "700"))
VAD_MAX_SILENCE_MS = float(os.getenv("VAD_MAX_SILENCE_MS", "3000"))

# Smart Turn v4 — semantic end-of-turn (optional refinement over the timeout).
SMART_TURN_ENABLED = os.getenv("SMART_TURN_ENABLED", "false").lower() in ("1", "true", "yes")
SMART_TURN_MODEL_PATH = _resolve_path(
    os.getenv("SMART_TURN_MODEL_PATH"), "models", "smart_turn_v4.onnx"
)
SMART_TURN_INPUT_NAME = os.getenv("SMART_TURN_INPUT_NAME", "")  # "" -> first model input
SMART_TURN_THRESHOLD = float(os.getenv("SMART_TURN_THRESHOLD", "0.5"))
SMART_TURN_MAX_SECONDS = float(os.getenv("SMART_TURN_MAX_SECONDS", "16"))

# --------------------------------------------------------------------------- #
# Export store — the Firestore-shaped audit documents.
# --------------------------------------------------------------------------- #
DB_PATH = _resolve_path(os.getenv("DB_PATH"), "data", "audits.db")

# --------------------------------------------------------------------------- #
# CORS: origins allowed to call this server (the separate Next dev frontend).
# --------------------------------------------------------------------------- #
CORS_ORIGINS = _split(
    os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:3001,http://localhost:3001,"
        "http://127.0.0.1:3000,http://localhost:3000",
    )
)

# --------------------------------------------------------------------------- #
# Static UI (produced by `npm run build:ui` from the sibling nextjs-app/ dir into ui-dist).
# --------------------------------------------------------------------------- #
UI_DIST_DIR = _resolve_path(os.getenv("UI_DIST_DIR"), "ui-dist")

# --------------------------------------------------------------------------- #
# LLM layer — see llm.py. Base URLs and default models resolved here so they are
# documented in one place; API keys are read lazily in llm.py (so a running
# server / a test can change them without re-import).
# --------------------------------------------------------------------------- #
DEFAULT_PROVIDER = "anthropic"

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5"

# Vision quality knobs. These apply only to models that accept them (Opus
# 4.6+/Sonnet 4.6+); on Haiku they are ignored, and llm.py sends `temperature`
# instead. Note Opus 4.7+ added high-resolution vision and materially better
# natural-image bounding-box localization/detection — measurably relevant here:
# on these showroom photos Haiku missed a shattered screen that Opus 4.8 found
# in both photos at 0.83-0.93 confidence. Set ANTHROPIC_EFFORT="" to disable
# adaptive thinking entirely.
ANTHROPIC_EFFORT = os.getenv("ANTHROPIC_EFFORT", "high")
ANTHROPIC_THINKING_MAX_TOKENS = int(os.getenv("ANTHROPIC_THINKING_MAX_TOKENS", "16000"))
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _base_url(name: str, default: str) -> str:
    return (os.getenv(name) or default).rstrip("/")


ANTHROPIC_BASE_URL = _base_url("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
OPENAI_BASE_URL = _base_url("OPENAI_BASE_URL", "https://api.openai.com/v1")
# Gemini's OpenAI-compatible endpoint.
GEMINI_BASE_URL = _base_url(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
)
VLLM_BASE_URL = _base_url("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
OLLAMA_BASE_URL = _base_url("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")

ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "") or "2023-06-01"
