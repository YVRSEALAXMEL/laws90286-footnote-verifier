"""
End-to-end test of the whole app with a SCRIPTED stand-in for OpenAI (no key, no cost).

Streamlit's AppTest runs home.py headless. We replace only the OpenAI client
(not ask_ai itself), so the real call counter, logging, JSON parsing and
schemas are exercised. The fake model's verdicts are crude keyword rules, so
this tests the plumbing and the CODE rules, not legal accuracy.

    python -m pytest -q tests/test_pipeline_offline.py
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import reusable_code.ai_client as ai_client
import reusable_code.pipeline as pipeline
from tests.fake_openai import FakeClient, FakeEmbeddings

HOME = Path(__file__).resolve().parent.parent / "home.py"


@pytest.fixture()
def fake_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(ai_client, "get_client", lambda: client)
    monkeypatch.setattr(pipeline, "make_embedding_function", lambda key: FakeEmbeddings())
    return client


def _run_test_case(label: str) -> AppTest:
    at = AppTest.from_file(str(HOME), default_timeout=180)
    at.session_state["openai_api_key"] = "sk-test-not-a-real-key"
    at.run()
    at.radio[0].set_value("Load a bundled test case").run()
    at.selectbox[0].set_value(label).run()
    next(b for b in at.button if b.label == "Run verification").click().run()
    assert not at.exception, at.exception
    return at


def _by_unit(at):
    return {r["unit_id"]: r for r in at.session_state["run"]["results"]}


def test_page_stops_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("reusable_code.api_key.os.getenv", lambda name, default=None: None)
    at = AppTest.from_file(str(HOME), default_timeout=60).run()
    assert not at.exception
    assert any("Enter your OpenAI API key" in i.value for i in at.info)
    assert not at.button or all(b.label != "Run verification" for b in at.button)
    # The key is never printed on the page.
    at.session_state["openai_api_key"] = "sk-secret-123"
    at.run()
    assert "sk-secret-123" not in str([e.value for e in at.markdown] + [e.value for e in at.success])


def test_tc1_plumbing(fake_client):
    at = _run_test_case("TC1 · Accurate citations (strengths)")
    run = at.session_state["run"]
    results = _by_unit(at)
    assert run["ai_call_count"] >= 5                       # brief: at least 5 AI calls per run
    assert run["ai_call_count"] == len(fake_client.requests)
    assert len(results) == 7
    # Ibid and (n X) resolve to the right files, deterministically.
    assert results["2.1"]["source_id"].startswith("Brown") and results["3.1"]["source_id"].startswith("Brown")
    assert results["4.1"]["source_id"].startswith("Brown")
    assert results["7.1"]["source_id"].startswith("Kapua")
    assert results["6.1"]["source_id"].startswith("R_v_Lojovic")
    # Ibid with a new pinpoint keeps its own pinpoint.
    assert results["2.1"]["pinpoint_text"] == "[6], [29]"
    # Every quotation in TC1 is found in its source.
    assert all(q["found"] for r in results.values() for q in r["quote_results"])


def test_tc2_code_rules(fake_client):
    results = _by_unit(_run_test_case("TC2 · Seeded citation errors"))
    # fn 4: fabricated quotation -> UNSUPPORTED by rule 3, whatever the AI says.
    assert results["4.1"]["verdict"] == "UNSUPPORTED" and results["4.1"]["rule"].startswith("R3")
    # fn 5: Kapua's words cited to Brown -> UNSUPPORTED, and the tool says where they really are.
    assert results["5.1"]["verdict"] == "UNSUPPORTED"
    elsewhere = results["5.1"]["quote_results"][0]["found_elsewhere"][0]
    assert elsewhere["source_id"].startswith("Kapua") and elsewhere["paras"] == [113]
    # fn 3: quote is at [111], cited [113] -> the pinpoint check flags it.
    assert results["3.1"]["pinpoint_check"]["status"] == "inconsistent"
    assert results["3.1"]["pinpoint_check"]["correct_paras"] == [111]
    # Code-decided verdicts skip the reviewer call (2 calls: plan + judge).
    assert len(results["4.1"]["turns"]) == 1 and results["4.1"]["turns"][0]["review"] is None


def test_tc3_limitations(fake_client):
    results = _by_unit(_run_test_case("TC3 · Limitations"))
    # fn 2: Verdins is not uploaded -> CANNOT_DETERMINE, with no AI judging from memory.
    assert results["2.1"]["verdict"] == "CANNOT_DETERMINE" and results["2.1"]["judgement"] is None
    # fn 3: a VR page pinpoint against a paragraph-numbered copy -> unverifiable, not wrong.
    assert results["3.1"]["pinpoint_check"]["status"] == "unverifiable"
    assert results["3.1"]["verdict"] != "WRONG_PINPOINT"
    # fn 4: two sources; the Strawson article is not provided.
    assert results["4.1"]["source_id"].startswith("Brown")
    assert results["4.2"]["verdict"] == "CANNOT_DETERMINE"
    # fn 1: the TC3(1) limitation - 'exceptional' IS "found" in Brown [6] (inside Jade's cited-by block).
    assert results["1.1"]["quote_results"][0]["found"] and 6 in results["1.1"]["quote_results"][0]["paras"]


def test_report_downloads(fake_client):
    from reusable_code.report import to_csv, to_docx
    run = _run_test_case("TC2 · Seeded citation errors").session_state["run"]
    assert to_csv(run["results"]).decode("utf-8-sig").startswith("fn,proposition")
    assert to_docx(run)[:2] == b"PK"   # a .docx is a zip file


def test_transient_error_is_retried_once_and_persistent_failure_is_contained(monkeypatch, fake_client):
    """A network blip is retried once; a citation whose AI call keeps failing is recorded, not fatal."""
    import httpx
    from openai import APIConnectionError
    monkeypatch.setattr(ai_client, "RETRY_WAIT_SECONDS", 0)
    real_create = fake_client.responses.create
    state = {"judge_calls": 0}

    def flaky_create(**kwargs):
        if "who_is_speaking" in kwargs["text"]["format"]["schema"]["properties"]:
            state["judge_calls"] += 1
            # First judge call: one blip (retried). fn 6 (Lojovic): always fails.
            if state["judge_calls"] == 1 or "Lojovic" in kwargs["input"]:
                raise APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
        return real_create(**kwargs)

    monkeypatch.setattr(fake_client.responses, "create", flaky_create)
    at = _run_test_case("TC1 · Accurate citations (strengths)")
    results = _by_unit(at)
    assert results["1.1"]["verdict"] != "CANNOT_DETERMINE"        # the blip was retried
    assert results["6.1"]["rule"] == "AI step failed"            # contained, run finished
    log = at.session_state["run"]["call_log"]
    assert sum(1 for c in log if not c["ok"]) == 3                # 1 blip + 2 attempts for fn 6


def test_password_gate():
    """With APP_PASSWORD set in Streamlit secrets, nothing else shows until it is entered."""
    at = AppTest.from_file(str(HOME), default_timeout=60)
    at.secrets["APP_PASSWORD"] = "test-pass"
    at.run()
    assert not at.exception
    assert at.title[0].value.startswith("🔒")
    assert not any(b.label == "Run verification" for b in at.button)
    at.text_input[0].input("wrong")
    at.button[0].click().run()
    assert any("Incorrect password" in e.value for e in at.error)
    at.text_input[0].input("test-pass")
    at.button[0].click().run()
    assert at.session_state["password_ok"] is True
    assert not at.title[0].value.startswith("🔒")
