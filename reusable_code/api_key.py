"""
API KEY SIDEBAR MODULE (required by the assignment brief)

Adapted from the LAWS90286 class credentials module:
    https://github.com/stoneman-mls/openai_credentials

It keeps the original's structure - render_sidebar() returns
(api_key, update_button) and the key lives in st.session_state.openai_api_key -
with these fixes:

  1. The text box uses type="password", so the key is masked on screen.
  2. The original line  st.write(f"Current API Key: {...}")  printed the key
     on the page. It is removed; we show "API key set ✓" instead.
  3. The original ran at import time. Python imports a module only once per
     process, so on Streamlit reruns the sidebar would vanish. Here it is a
     function that every page calls on every run.
  4. require_api_key() stops the page (st.stop()) with a friendly message
     until a key has been entered.
  5. For LOCAL development only, a key in .env (OPENAI_API_KEY) is accepted
     as a fallback. On Streamlit Cloud there is no .env (it is in .gitignore)
     and no secret is configured, so users must enter their own key.
"""

import os

import streamlit as st


def render_sidebar():
    st.sidebar.title("API Key Input")
    api_key = st.sidebar.text_input("Enter your API key:", type="password",
                                    help="Your OpenAI key is kept only in this browser session and is never stored or shown.")
    update_button = st.sidebar.button("Update API Key")
    return api_key, update_button


def api_key_module():
    """Draw the sidebar module and store the key in session state."""
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = None

    api_key, update_button = render_sidebar()
    if update_button:
        st.session_state.openai_api_key = api_key.strip() or None

    # Fix 2: never print the key itself.
    if st.session_state.openai_api_key:
        st.sidebar.success("API key set ✓")
    elif os.getenv("OPENAI_API_KEY"):
        st.sidebar.info("Using OPENAI_API_KEY from your local .env file (development only).")
    else:
        st.sidebar.warning("No API key set yet.")


def current_api_key():
    """The key to use: the user's session key first, then the local .env fallback."""
    return st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")


def require_api_key():
    """Stop the page until a key is available (Fix 4)."""
    if not current_api_key():
        st.info("👈 Enter your OpenAI API key in the sidebar and click **Update API Key** to start. "
                "The key stays in your browser session; this app never stores it.")
        st.stop()
