"""
A SCRIPTED STAND-IN for the OpenAI API, used only by tests/test_pipeline_offline.py.

It lets the whole pipeline run with no API key and no cost, to test the
PLUMBING (schemas, call counting, Ibid / (n X) resolution, the code-based
grounding rules, the reflection loop, the reports). Its "judgements" are
crude keyword rules - they say nothing about how well the real model performs.
Real results come from running the app with a key (see README, Test cases).

Every request is checked: the JSON schema must satisfy OpenAI's strict-mode
rules, and every fake reply must validate against the schema it was given.
"""

import hashlib
import json
import re
from types import SimpleNamespace

import jsonschema
import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

QUOTE = re.compile(r"(?<![A-Za-z])‘(.+?)’(?![A-Za-z])")


def assert_strict(schema: dict, path: str = "$"):
    """OpenAI strict mode: every object lists ALL its properties as required and forbids extras."""
    if schema.get("type") == "object" or "properties" in schema:
        assert schema.get("additionalProperties") is False, f"{path}: additionalProperties must be false"
        assert set(schema.get("required", [])) == set(schema["properties"]), f"{path}: all properties must be required"
        for name, sub in schema["properties"].items():
            assert_strict(sub, f"{path}.{name}")
    if "items" in schema:
        assert_strict(schema["items"], f"{path}[]")


class FakeEmbeddings(EmbeddingFunction):
    """Bag-of-words hashing 'embeddings' so Chroma can run offline."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            v = np.zeros(256)
            for word in re.findall(r"[a-z]{3,}", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
            vectors.append(v / (np.linalg.norm(v) or 1))
        return vectors

    @staticmethod
    def name():
        return "fake-bag-of-words"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return FakeEmbeddings()


# ---------------------------------------------------------------------------
# One handler per step, chosen by the schema's property names
# ---------------------------------------------------------------------------
def _identify(prompt):
    filename = re.search(r"File name: (.+)", prompt).group(1)
    mnc = re.search(r"\[\d{4}\] [A-Z]+ \d+|\d{4} ONCA \d+", prompt)
    name = {"Brown": "Brown v The Queen", "Kapua": "Kapua v R", "R_v_Lojovic": "R v Lojovic"}
    case = next((v for k, v in name.items() if filename.startswith(k)), filename)
    return {"case_name": case, "medium_neutral_citation": mnc.group(0) if mnc else None, "reported_citations": [],
            "court": "Court of Appeal", "decision_date": "", "judges": [], "document_type": "judgment",
            "paragraph_numbering_style": "N.", "short_title": case.split()[0] if case != "R v Lojovic" else "Lojovic",
            "notes": ""}


def _claims(prompt):
    body = prompt.split("Footnotes to process:")[0]
    claims = []
    for m in re.finditer(r"\[\^(\d+)\]", body):
        before = re.sub(r"\[\^\d+\]", "|", body[:m.start()]).split("|")[-1]
        sentence = before.strip().split("\n")[-1]
        attributed = "the Court of Appeal (holding)" if "Court of Appeal" in sentence and "held" in sentence else "the source"
        claims.append({"fn": int(m.group(1)), "sentence": sentence, "proposition": sentence,
                       "attributed_to": attributed, "quoted_phrases": QUOTE.findall(sentence),
                       "authorial_comment": "language close to Strawson's invitation" if "Strawson" in sentence else ""})
    return {"claims": claims}


def _resolve(prompt, schema):
    ids = schema["properties"]["footnotes"]["items"]["properties"]["units"]["items"]["properties"]["matched_source_id"]["enum"]
    def match(text):
        for word, prefix in [("Brown", "Brown"), ("Kapua", "Kapua"), ("Lojovic", "R_v_Lojovic")]:
            if word in text:
                return next((i for i in ids if i.startswith(prefix)), "NOT_PROVIDED")
        return "NOT_PROVIDED"
    footnotes = []
    for m in re.finditer(r"^Footnote (\d+): (.+)$", prompt, re.M):
        fn, text = int(m.group(1)), m.group(2)
        units = []
        for part in [p.strip() for p in text.split(";")]:
            pin = re.search(r"((?:\[\d+\](?:[–-]\[\d+\])?(?:,\s*)?)+|\b\d{3}(?:[–-]\d+)?)\.?$", part.rstrip("."))
            ref = "ibid" if part.lower().startswith("ibid") else "cross_reference" if "(n " in part else "full"
            units.append({"citation_text": part, "reference_type": ref, "refers_to_fn": None,
                          "source_kind": "secondary" if "Strawson" in part else "case",
                          "case_name": part.split("(")[0].split("[")[0].strip(), "full_citation": part,
                          "pinpoint_text": pin.group(1) if pin else "", "matched_source_id": match(part),
                          "match_reason": "fake", "unit_proposition": "",
                          "unit_quoted_phrases": []})
        footnotes.append({"fn": fn, "units": units})
    return {"footnotes": footnotes}


def _judge(prompt):
    excerpts = prompt.split("EXCERPTS (labelled with paragraph number and how each was found):")[1]
    excerpts = excerpts.split("WORK RECORD OF EARLIER TURNS:")[0]
    blocks = re.findall(r"^\[(?:[^\[\]]|\[\d+\])+\]\n(.+?)(?=\n\n\[|\Z)", excerpts, re.S | re.M)
    labels = re.findall(r"^\[[^\[]+\[(\d+)\]", excerpts, re.M)
    proposition = re.search(r"PROPOSITION THIS SOURCE IS CITED FOR: (.*)", prompt).group(1)
    attributed = re.search(r"DRAFT ATTRIBUTES IT TO: (.*)", prompt).group(1)
    phrases = re.findall(r"‘(.+?)’ - computer check: found", prompt)
    best, para = "", None
    for label, block in zip(labels, blocks):
        sentence = block.split(". ")[0].split("\n")[0][:300]
        if not phrases or any(p.lower() in block.lower() for p in phrases):
            best, para = sentence, int(label)
            break
    content = "full" if best else "none"
    strength = "draft_stronger" if re.search(r"\bnever\b", proposition) else "same"
    attribution = ("different" if "Court of Appeal" in attributed and "trial judge" in " ".join(blocks).lower()
                   else "same")
    verdict = ("UNSUPPORTED" if content == "none" else "MISATTRIBUTED" if attribution == "different"
               else "OVERSTATED" if strength == "draft_stronger" else "SUPPORTED")
    return {"draft_attribution": attributed, "who_is_speaking": "court_or_majority", "speaker_detail": "fake",
            "content_match": content, "attribution_match": attribution if content != "none" else "not_applicable",
            "strength_match": strength, "found_at_para": para, "supporting_quote": best, "verdict": verdict,
            "reasoning": "scripted fake", "confidence": 0.9}


def _reply(kwargs):
    schema = kwargs["text"]["format"]["schema"]
    keys = set(schema["properties"])
    prompt = kwargs["input"]
    if "case_name" in keys and "judges" in keys:
        return _identify(prompt)
    if keys == {"claims"}:
        return _claims(prompt)
    if keys == {"footnotes"}:
        return _resolve(prompt, schema)
    if keys == {"queries"}:
        return {"queries": [re.search(r"Proposition the source is cited for: (.*)", prompt).group(1)[:80] or "reasons"]}
    if "who_is_speaking" in keys:
        return _judge(prompt)
    if "verdict_well_founded" in keys:
        return {"verdict_well_founded": True, "problems": [], "search_next": [], "reviewer_verdict": "SUPPORTED"}
    if "fix_type" in keys:
        return {"fix_type": "soften_wording", "suggested_sentence": "", "suggested_footnote": "",
                "explanation": "scripted fake"}
    if "overall_assessment" in keys:
        return {"overall_assessment": "Scripted summary.", "most_serious_issues": [], "next_steps": []}
    raise AssertionError(f"Unknown schema: {keys}")


class FakeResponses:
    def __init__(self, log):
        self.log = log

    def create(self, **kwargs):
        assert kwargs["model"] and kwargs["reasoning"]["effort"] in ("low", "medium", "high", "xhigh", "max")
        schema = kwargs["text"]["format"]["schema"]
        assert kwargs["text"]["format"]["strict"] is True
        assert_strict(schema)
        reply = _reply(kwargs)
        jsonschema.validate(reply, schema)
        self.log.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(reply),
                               usage=SimpleNamespace(input_tokens=len(kwargs["input"]) // 4, output_tokens=200))


class FakeClient:
    def __init__(self):
        self.requests = []
        self.responses = FakeResponses(self.requests)
