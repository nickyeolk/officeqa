"""Shared configuration for the OfficeQA agent-harness eval."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from this directory regardless of cwd
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# Quiet noisy third-party logs unless the user opted in.
os.environ.setdefault("LITELLM_LOG", "ERROR")
os.environ.setdefault("LITELLM_SUPPRESS_DEBUG_INFO", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

# --- paths ---
EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
CORPUS_DIR = (PROJECT_ROOT / "treasury_bulletins_parsed" / "transformed").resolve()
RESULTS_DIR = EVAL_DIR / "results"
CSV_PRO = PROJECT_ROOT / "officeqa_pro.csv"
CSV_FULL = PROJECT_ROOT / "officeqa_full.csv"

# --- OpenRouter ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# --- Self-hosted endpoint (vLLM / LiteLLM behind Cloudflare Access) ---
# Full URL as stored in .env, e.g. https://host/v1/chat/completions
_HOSTED_ENDPOINT_RAW = os.environ.get("HOSTED_ENDPOINT", "")
CF_CLIENT_ID = os.environ.get("CF_ACCESS_CLIENT_ID", "")
CF_CLIENT_SECRET = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")


def get_hosted_base_url() -> str:
    """Return the base URL (strip trailing /chat/completions if present)."""
    url = _HOSTED_ENDPOINT_RAW.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    return url


def get_cf_headers() -> dict[str, str]:
    """Return Cloudflare Access service-token headers (empty dict if not configured)."""
    headers: dict[str, str] = {}
    if CF_CLIENT_ID:
        headers["CF-Access-Client-Id"] = CF_CLIENT_ID
    if CF_CLIENT_SECRET:
        headers["CF-Access-Client-Secret"] = CF_CLIENT_SECRET
    return headers


def require_hosted_base_url() -> str:
    """Return the hosted base URL or raise a clear error if not configured."""
    url = get_hosted_base_url()
    if not url:
        raise RuntimeError(
            "HOSTED_ENDPOINT is not set. "
            f"Put it in {_ENV_PATH} (see .env.example)."
        )
    return url


# --- default models (override via env vars) ---
# As of May 2026 — adjust to whatever OpenRouter has live when you run.
# See https://openrouter.ai/models for current IDs.
# Model name as served by vLLM (check GET /v1/models if unsure).
# The underlying weights are Qwen/Qwen3.6-35B-A3B but vLLM registers it
# under the alias defined at startup — "qwen-local" in this deployment.
_HOSTED_MODEL_DEFAULT = os.environ.get("HOSTED_MODEL", "qwen-local")
DEFAULT_MODELS: dict[str, str] = {
    "claude": os.environ.get("CLAUDE_MODEL", "anthropic/claude-opus-4"),
    "openai": os.environ.get("OPENAI_MODEL", "openai/gpt-5"),
    "google": os.environ.get("GOOGLE_MODEL", "google/gemini-2.5-pro"),
    "microsoft": os.environ.get("MICROSOFT_MODEL", "openai/gpt-5"),
    # hosted runners — all default to the same self-hosted model
    "hosted_openai": _HOSTED_MODEL_DEFAULT,
    "hosted_microsoft": _HOSTED_MODEL_DEFAULT,
    "hosted_google": _HOSTED_MODEL_DEFAULT,
}

# --- agent budget ---
MAX_TURNS = 30  # max agent turns per question (prevents runaway)
RETRY_MAX = 3
RETRY_BACKOFF_S = 2.0
DEFAULT_TOLERANCE = 0.01  # 1 % — matches paper headline


def require_openrouter_key() -> str:
    """Fetch the OpenRouter key or raise a clear error."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            f"Put it in {_ENV_PATH} (see .env.example)."
        )
    return OPENROUTER_API_KEY
