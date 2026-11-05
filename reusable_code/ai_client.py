"""
AI HELPER - ask_ai(), based on the day 5 class helper (Responses API with a
strict JSON schema), extended to:

  (a) build the OpenAI client INSIDE the function from the user's session key
      (the class version created OpenAI() at import time from .env, which
      cannot work on Streamlit Cloud where each user brings their own key);
  (b) count every AI call in st.session_state and log it (step name, model,
      duration, tokens), so the app can show "AI calls this run: N"; and
  (c) retry ONCE on a transient error (network blip, rate limit, server error).
"""

import json
import time

import streamlit as st
from openai import (APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError,
                    InternalServerError, OpenAI, RateLimitError)

from reusable_code.api_key import current_api_key

DEFAULT_MODEL = "gpt-6.1-sol"
EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"]   # as in the class helper
TRANSIENT_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)
RETRY_WAIT_SECONDS = 3


class AIError(RuntimeError):
    """A friendly error for the UI."""


def get_client() -> OpenAI:
    """Create the client from the CURRENT user's key (never at import time)."""
    key = current_api_key()
    if not key:
        raise AIError("No OpenAI API key has been entered.")
    return OpenAI(api_key=key, timeout=180, max_retries=0)   # we do our own single retry


# ---------------------------------------------------------------------------
# Call counter and log (kept in session state so the UI can display them)
# ---------------------------------------------------------------------------
def reset_call_log():
    st.session_state.ai_call_count = 0
    st.session_state.ai_call_log = []
    st.session_state.embedding_requests = 0


def _log(step, model, seconds, ok, usage=None, error=""):
    st.session_state.setdefault("ai_call_log", [])
    st.session_state.setdefault("ai_call_count", 0)
    if ok:
        st.session_state.ai_call_count += 1
    st.session_state.ai_call_log.append({
        "n": st.session_state.ai_call_count if ok else None,
        "step": step,
        "model": model,
        "seconds": round(seconds, 1),
        "ok": ok,
        "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        "error": error,
    })


def log_embedding_requests(n: int = 1):
    """Embedding requests are counted separately; they are not 'AI calls' to the reasoning model."""
    st.session_state.embedding_requests = st.session_state.get("embedding_requests", 0) + n


# ---------------------------------------------------------------------------
# The helper itself
# ---------------------------------------------------------------------------
def ask_ai(prompt: str, instructions: str | None = None, effort: str = "medium", schema: dict | None = None,
           model: str | None = None, step: str = "unnamed step"):
    """
    One call to the Responses API. With a schema, the reply is forced to match
    it (strict JSON) and is returned as a Python dict; otherwise returns text.
    """
    model = model or st.session_state.get("settings", {}).get("model", DEFAULT_MODEL)
    kwargs = {"model": model, "input": prompt, "reasoning": {"effort": effort}}
    if instructions:
        kwargs["instructions"] = instructions
    if schema:
        kwargs["text"] = {"format": {"type": "json_schema", "name": "output", "strict": True, "schema": schema}}

    client = get_client()
    for attempt in (1, 2):
        start = time.time()
        try:
            response = client.responses.create(**kwargs)
            result = json.loads(response.output_text) if schema else response.output_text
            _log(step, model, time.time() - start, ok=True, usage=getattr(response, "usage", None))
            return result
        except AuthenticationError as error:
            _log(step, model, time.time() - start, ok=False, error="authentication failed")
            raise AIError("OpenAI rejected the API key. Check it in the sidebar and click Update API Key.") from error
        except (TRANSIENT_ERRORS + (json.JSONDecodeError,)) as error:
            _log(step, model, time.time() - start, ok=False, error=f"{type(error).__name__} (attempt {attempt})")
            if attempt == 2:
                raise AIError(f"The AI call for '{step}' failed twice: {type(error).__name__}: {error}") from error
            time.sleep(RETRY_WAIT_SECONDS)
        except APIStatusError as error:
            # e.g. an unknown model name or an invalid request: retrying would not help.
            _log(step, model, time.time() - start, ok=False, error=f"{type(error).__name__}")
            raise AIError(f"OpenAI refused the request for '{step}' (model '{model}'): {error}") from error
