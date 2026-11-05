"""
SIMPLE APP PASSWORD (optional)

The deployed app is public on Streamlit Community Cloud, so a short shared
password keeps casual visitors out. The password is NOT in the code or the
repository: it is set in the Streamlit Cloud app settings as a secret

    APP_PASSWORD = "..."

When no APP_PASSWORD secret exists (for example in local development), the
gate is switched off and the app opens straight away.

This is a convenience lock, not real security: the code itself is on GitHub,
and every user still needs their own OpenAI API key to run a check.
"""

import hmac

import streamlit as st


def _configured_password():
    try:
        return st.secrets.get("APP_PASSWORD")
    except Exception:   # no secrets.toml at all (local development)
        return None


def require_password():
    """Show a password form and stop the page until the right password is entered."""
    expected = _configured_password()
    if not expected or st.session_state.get("password_ok"):
        return
    st.title("🔒 Footnote–Source Verifier")
    st.write("This app is password-protected. The password is given with the assignment submission.")
    with st.form("password_form"):
        entered = st.text_input("App password", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        # compare_digest avoids leaking how many characters matched (good habit for any secret check)
        if hmac.compare_digest(entered.encode(), str(expected).encode()):
            st.session_state.password_ok = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()
