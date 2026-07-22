"""Server-side LLM transport for the fixture-audit routes.

Python port of lib/merchandise/llm.ts, extended with per-role configuration and
two more providers:

  Providers: anthropic (alias: claude), openai (alias: chatgpt),
             gemini (alias: google), vllm (alias: local), ollama.

  Roles:     "vision" — used by /api/merchandise/analyze
             "chat"   — used by /api/merchandise/voice-turn

Per-role env resolution: VISION_LLM_PROVIDER / VISION_LLM_MODEL and
CHAT_LLM_PROVIDER / CHAT_LLM_MODEL, each falling back to LLM_PROVIDER /
LLM_MODEL (so both roles can share one model), with legacy AI_PROVIDER as the
final provider fallback. Default provider: anthropic.

Two transports, exactly like llm.ts: Anthropic native /messages, and OpenAI
/chat/completions for the openai / gemini / vllm / ollama providers (Gemini via
its OpenAI-compatible endpoint).
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

import httpx

from . import config

PROVIDERS = ("anthropic", "openai", "gemini", "vllm", "ollama")

# Generous timeouts — local vision models are slow.
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)

# Auto-discovered model ids for the local providers, keyed by base URL.
_cached_local_models: dict[str, str] = {}


class LlmError(Exception):
    """Error with the same metadata shape the TS transport attaches:
    code, provider, status, detail, baseUrl."""

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        provider: Optional[str] = None,
        status: Optional[int] = None,
        detail: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.provider = provider
        self.status = status
        self.detail = detail
        self.base_url = base_url


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def normalize_provider(value: str) -> Optional[str]:
    normalized = (value or "").strip().lower()
    if not normalized:
        return None
    if normalized in ("claude", "anthropic"):
        return "anthropic"
    if normalized in ("openai", "chatgpt"):
        return "openai"
    if normalized in ("gemini", "google"):
        return "gemini"
    if normalized in ("local", "vllm"):
        return "vllm"
    if normalized == "ollama":
        return "ollama"
    return None


def _role_prefix(role: str) -> str:
    return "VISION" if role == "vision" else "CHAT"


def get_configured_provider(role: str) -> str:
    """Resolve the provider for a role: {ROLE}_LLM_PROVIDER -> LLM_PROVIDER ->
    legacy AI_PROVIDER -> default anthropic."""
    raw = (
        _env(f"{_role_prefix(role)}_LLM_PROVIDER")
        or _env("LLM_PROVIDER")
        or _env("AI_PROVIDER")
        or config.DEFAULT_PROVIDER
    )
    provider = normalize_provider(raw)
    if not provider:
        raise LlmError(
            f'Invalid LLM_PROVIDER "{raw}".',
            code="invalid_provider",
            provider=raw,
            detail=(
                "Use LLM_PROVIDER=anthropic, LLM_PROVIDER=openai, "
                "LLM_PROVIDER=gemini, LLM_PROVIDER=vllm, or LLM_PROVIDER=ollama."
            ),
        )
    return provider


def base_url_for_provider(provider: str) -> str:
    if provider == "anthropic":
        return config.ANTHROPIC_BASE_URL
    if provider == "openai":
        return config.OPENAI_BASE_URL
    if provider == "gemini":
        return config.GEMINI_BASE_URL
    if provider == "ollama":
        return config.OLLAMA_BASE_URL
    return config.VLLM_BASE_URL


def llm_provider_label(provider: str) -> str:
    if provider in ("anthropic", "claude"):
        return "Claude"
    if provider == "openai":
        return "OpenAI"
    if provider == "gemini":
        return "Gemini"
    if provider in ("vllm", "local"):
        return "local vLLM"
    if provider == "ollama":
        return "Ollama"
    return "LLM"


def _env_model(role: str) -> str:
    """The explicit per-role model from the environment ({ROLE}_LLM_MODEL ->
    LLM_MODEL)."""
    return _env(f"{_role_prefix(role)}_LLM_MODEL") or _env("LLM_MODEL")


def _legacy_env_model(provider: str) -> str:
    """Legacy single-provider model overrides kept from the TS app."""
    if provider == "anthropic":
        return _env("CLAUDE_MODEL")
    if provider == "openai":
        return _env("OPENAI_MODEL")
    if provider == "gemini":
        return _env("GEMINI_MODEL")
    if provider == "ollama":
        return _env("OLLAMA_MODEL")
    return _env("VLLM_MODEL")


def _default_model(provider: str) -> str:
    if provider == "anthropic":
        return config.DEFAULT_CLAUDE_MODEL
    if provider == "openai":
        return config.DEFAULT_OPENAI_MODEL
    if provider == "gemini":
        return config.DEFAULT_GEMINI_MODEL
    return ""  # vllm / ollama auto-discover


async def _resolve_local_model(provider: str, explicit: str) -> str:
    """Auto-discover the served model for vllm/ollama via GET {base_url}/models.

    Mirrors resolveLocalModel() in llm.ts: explicit model wins, then the legacy
    env override, then a cached discovery, then the first /models entry. Error
    codes: local_unreachable, local_models_error, local_model_missing.
    """
    if explicit:
        return explicit
    legacy = _legacy_env_model(provider)
    if legacy:
        return legacy

    base_url = base_url_for_provider(provider)
    cached = _cached_local_models.get(base_url)
    if cached:
        return cached

    label = llm_provider_label(provider)
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(f"{base_url}/models")
    except Exception as exc:  # noqa: BLE001 - shaped into the route error
        raise LlmError(
            f"Could not reach {label} /models.",
            code="local_unreachable",
            provider=provider,
            detail=str(exc),
            base_url=base_url,
        ) from exc
    if response.status_code >= 400:
        try:
            detail = response.text
        except Exception:  # noqa: BLE001
            detail = ""
        raise LlmError(
            f"{label} /models returned {response.status_code}",
            code="local_models_error",
            provider=provider,
            status=response.status_code,
            detail=detail,
            base_url=base_url,
        )
    try:
        data = response.json()
    except Exception:  # noqa: BLE001
        data = None
    model_id = None
    if isinstance(data, dict):
        entries = data.get("data")
        if isinstance(entries, list) and entries and isinstance(entries[0], dict):
            model_id = entries[0].get("id")
    if not isinstance(model_id, str) or not model_id:
        raise LlmError(
            f"{label} is running but is not serving any model.",
            code="local_model_missing",
            provider=provider,
            base_url=base_url,
        )
    _cached_local_models[base_url] = model_id
    return model_id


async def resolve_llm(role: str, requested_model: Optional[str] = None) -> dict:
    """Resolve {provider, model, baseUrl} for a role.

    ``requested_model`` (e.g. the request-body ``model``) wins over the env.
    """
    provider = get_configured_provider(role)
    explicit = (requested_model or "").strip() or _env_model(role)

    if provider in ("vllm", "ollama"):
        model = await _resolve_local_model(provider, explicit)
    else:
        model = explicit or _legacy_env_model(provider) or _default_model(provider)

    return {"provider": provider, "model": model, "baseUrl": base_url_for_provider(provider)}


def fallback_llm(provider: str, requested_model: Optional[str] = None, role: str = "chat") -> dict:
    """Best-effort resolution with no network (port of fallbackLlm in llm.ts).

    For local providers the model may be empty when nothing is configured; the
    downstream completion call will then fail and be reported as parse_error.
    """
    explicit = (requested_model or "").strip() or _env_model(role)
    if provider in ("vllm", "ollama"):
        model = explicit or _legacy_env_model(provider)
    else:
        model = explicit or _legacy_env_model(provider) or _default_model(provider)
    return {"provider": provider, "model": model, "baseUrl": base_url_for_provider(provider)}


def voice_model_override(body_model: Optional[str], provider: str) -> Optional[str]:
    """The chat-role orchestrator model override chain from voice-turn/route.ts:
    request body model -> LLM_ORCHESTRATOR_MODEL -> per-provider legacy
    *_ORCHESTRATOR_MODEL."""
    if body_model:
        return body_model
    generic = _env("LLM_ORCHESTRATOR_MODEL")
    if generic:
        return generic
    if provider == "anthropic":
        return _env("CLAUDE_ORCHESTRATOR_MODEL") or None
    if provider == "openai":
        return _env("OPENAI_ORCHESTRATOR_MODEL") or None
    if provider == "gemini":
        return _env("GEMINI_ORCHESTRATOR_MODEL") or None
    return _env("VLLM_ORCHESTRATOR_MODEL") or None


def describe_llm(role: str) -> dict:
    """Safe, network-free description of the resolved provider+model for /health.

    Never raises: an invalid provider is reported as an ``error`` string. For
    the local providers with no configured model, reports the cached discovery
    if one exists, else "(auto-discover)".
    """
    try:
        provider = get_configured_provider(role)
    except LlmError as exc:
        return {"error": str(exc)}
    explicit = _env_model(role)
    if provider in ("vllm", "ollama"):
        model = (
            explicit
            or _legacy_env_model(provider)
            or _cached_local_models.get(base_url_for_provider(provider))
            or "(auto-discover)"
        )
    else:
        model = explicit or _legacy_env_model(provider) or _default_model(provider)
    return {"provider": provider, "model": model, "baseUrl": base_url_for_provider(provider)}


# --------------------------------------------------------------------------- #
# OpenAI-compatible transport (openai / gemini / vllm / ollama).
# --------------------------------------------------------------------------- #


def _extract_openai_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
            parts.append(part["text"])
        else:
            parts.append("")
    return "".join(parts)


def _api_key_for(provider: str) -> Optional[str]:
    """The bearer key for an OpenAI-compatible provider, raising missing_api_key
    where a key is required (openai, gemini). Local providers need none."""
    if provider == "openai":
        api_key = _env("OPENAI_API_KEY")
        if not api_key:
            raise LlmError(
                "Missing OPENAI_API_KEY for the OpenAI provider.",
                code="missing_api_key",
                provider="openai",
                base_url=config.OPENAI_BASE_URL,
            )
        return api_key
    if provider == "gemini":
        api_key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
        if not api_key:
            raise LlmError(
                "Missing GEMINI_API_KEY for the Gemini provider.",
                code="missing_api_key",
                provider="gemini",
                base_url=config.GEMINI_BASE_URL,
            )
        return api_key
    return None


async def _openai_compatible_chat_completion(
    *,
    provider: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    json_object: bool = False,
) -> str:
    base_url = base_url_for_provider(provider)
    headers = {"Content-Type": "application/json"}
    api_key = _api_key_for(provider)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_object:
        body["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(f"{base_url}/chat/completions", headers=headers, json=body)

    if response.status_code >= 400:
        try:
            detail = response.text
        except Exception:  # noqa: BLE001
            detail = ""
        raise LlmError(
            f"{llm_provider_label(provider)} returned {response.status_code}",
            code=f"{provider}_error",
            provider=provider,
            status=response.status_code,
            detail=detail,
            base_url=base_url,
        )

    data = response.json()
    content = ""
    if isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                content = _extract_openai_text(message.get("content"))
    if not content.strip():
        raise LlmError(
            f"{llm_provider_label(provider)} returned an empty message.",
            code=f"{provider}_empty_message",
            provider=provider,
            base_url=base_url,
        )
    return content


# --------------------------------------------------------------------------- #
# Anthropic native transport.
# --------------------------------------------------------------------------- #

_DATA_URL_RE = re.compile(r"^data:([^;,]+);base64,(.*)$", re.IGNORECASE | re.DOTALL)


def _content_to_system_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _data_url_to_anthropic_image(url: str) -> dict:
    match = _DATA_URL_RE.match(url or "")
    if not match:
        raise LlmError(
            "Claude vision input must be a base64 data URL.",
            code="invalid_image_url",
            provider="anthropic",
            base_url=config.ANTHROPIC_BASE_URL,
        )
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": match.group(1) or "image/jpeg",
            "data": re.sub(r"\s", "", match.group(2)),
        },
    }


def _to_anthropic_content(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    out = []
    for part in content:
        if part.get("type") == "text":
            out.append({"type": "text", "text": part.get("text", "")})
        else:
            out.append(_data_url_to_anthropic_image(part.get("image_url", {}).get("url", "")))
    return out


def _to_anthropic_messages(messages: list[dict]) -> dict:
    system: list[str] = []
    anthropic_messages: list[dict] = []
    for message in messages:
        if message.get("role") == "system":
            text = _content_to_system_text(message.get("content")).strip()
            if text:
                system.append(text)
            continue
        anthropic_messages.append(
            {"role": message["role"], "content": _to_anthropic_content(message.get("content"))}
        )
    return {"system": "\n\n".join(system), "messages": anthropic_messages}


# Claude 4.7+ removed the sampling parameters: sending `temperature` (or
# top_p/top_k) to these models is a 400, not a warning. The list is by prefix
# because these IDs carry no date suffix.
_NO_SAMPLING_PARAMS = (
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)

# Models accepting `thinking: {"type": "adaptive"}` + `output_config.effort`.
# On the 4.7+ models thinking is OFF unless asked for explicitly — omitting the
# field does not enable it.
_ADAPTIVE_THINKING = _NO_SAMPLING_PARAMS + ("claude-opus-4-6", "claude-sonnet-4-6")


def _model_is(model: str, families: tuple) -> bool:
    name = (model or "").strip()
    return any(name.startswith(f) for f in families)


async def _claude_chat_completion(
    *, model: str, messages: list[dict], temperature: float, max_tokens: int
) -> str:
    api_key = _env("ANTHROPIC_API_KEY") or _env("CLAUDE_API_KEY")
    if not api_key:
        raise LlmError(
            "Missing ANTHROPIC_API_KEY for the Claude provider.",
            code="missing_api_key",
            provider="anthropic",
            base_url=config.ANTHROPIC_BASE_URL,
        )

    converted = _to_anthropic_messages(messages)
    body: dict[str, Any] = {"model": model, "max_tokens": max_tokens}

    if not _model_is(model, _NO_SAMPLING_PARAMS):
        body["temperature"] = temperature

    if config.ANTHROPIC_EFFORT and _model_is(model, _ADAPTIVE_THINKING):
        body["thinking"] = {"type": "adaptive"}
        body["output_config"] = {"effort": config.ANTHROPIC_EFFORT}
        # Thinking and the visible answer share `max_tokens`. A budget sized
        # for a non-thinking reply gets consumed by reasoning and the JSON
        # payload truncates mid-object, so give it room.
        body["max_tokens"] = max(max_tokens, config.ANTHROPIC_THINKING_MAX_TOKENS)

    if converted["system"]:
        body["system"] = converted["system"]
    body["messages"] = converted["messages"]

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(
            f"{config.ANTHROPIC_BASE_URL}/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": config.ANTHROPIC_VERSION,
            },
            json=body,
        )

    if response.status_code >= 400:
        try:
            detail = response.text
        except Exception:  # noqa: BLE001
            detail = ""
        raise LlmError(
            f"Claude returned {response.status_code}",
            code="anthropic_error",
            provider="anthropic",
            status=response.status_code,
            detail=detail,
            base_url=config.ANTHROPIC_BASE_URL,
        )

    data = response.json()
    content = ""
    if isinstance(data, dict) and isinstance(data.get("content"), list):
        content = "".join(
            part["text"]
            for part in data["content"]
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
        )
    if not content.strip():
        raise LlmError(
            "Claude returned an empty message.",
            code="anthropic_empty_message",
            provider="anthropic",
            base_url=config.ANTHROPIC_BASE_URL,
        )
    return content


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #


async def chat_completion(
    *,
    provider: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    json_object: bool = False,
) -> str:
    """Run one chat completion. ``messages`` use the OpenAI shape:
    [{role, content}] where content is a string or a list of
    {type:"text", text} / {type:"image_url", image_url:{url}} parts.

    ``json_object`` constrains OpenAI-compatible providers to a single JSON
    object; it is ignored by the Anthropic native transport (like the TS)."""
    if provider == "anthropic":
        return await _claude_chat_completion(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
    return await _openai_compatible_chat_completion(
        provider=provider,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        json_object=json_object,
    )


def llm_setup_message(provider: str) -> str:
    if provider in ("anthropic", "claude"):
        return (
            "Claude is the default LLM provider. Set ANTHROPIC_API_KEY in python-server/.env, "
            "or switch providers with LLM_PROVIDER=openai, LLM_PROVIDER=gemini, "
            "LLM_PROVIDER=vllm, or LLM_PROVIDER=ollama."
        )
    if provider == "openai":
        return (
            "OpenAI is selected as the LLM provider. Set OPENAI_API_KEY in python-server/.env, "
            "or switch providers with LLM_PROVIDER=anthropic, LLM_PROVIDER=gemini, "
            "LLM_PROVIDER=vllm, or LLM_PROVIDER=ollama."
        )
    if provider == "gemini":
        return (
            "Gemini is selected as the LLM provider. Set GEMINI_API_KEY (or GOOGLE_API_KEY) "
            "in python-server/.env, or switch providers with LLM_PROVIDER=anthropic, "
            "LLM_PROVIDER=openai, LLM_PROVIDER=vllm, or LLM_PROVIDER=ollama."
        )
    if provider in ("vllm", "local"):
        return (
            f"Could not reach the vLLM server at {config.VLLM_BASE_URL}. Start it with "
            '"vllm serve <model> --host 0.0.0.0 --port 8000" and confirm it is reachable '
            "(set VLLM_BASE_URL if it binds elsewhere)."
        )
    if provider == "ollama":
        return (
            f"Could not reach the Ollama server at {config.OLLAMA_BASE_URL}. Start it with "
            '"ollama serve" and pull a model, and confirm it is reachable '
            "(set OLLAMA_BASE_URL if it binds elsewhere)."
        )
    return "Set LLM_PROVIDER to anthropic, openai, gemini, vllm, or ollama."
