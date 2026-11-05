"""
STEP 11 - SUMMARISE THE REPORT (1 AI call) + CSV and .docx exports (no AI)

The summary call sees only the per-citation results (not the sources), and
must put the most serious problems first: a quotation that is not in the
source is worse than an overstatement, which is worse than a pinpoint that
is one or two paragraphs out.
"""

import csv
import io
from datetime import datetime

from docx import Document
from docx.shared import Pt, RGBColor

from reusable_code.ai_client import ask_ai
from reusable_code.verdict_rules import SEVERITY, VERDICTS

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_assessment": {"type": "string"},
        "most_serious_issues": {"type": "array", "items": {
            "type": "object",
            "properties": {"fn": {"type": "string"}, "issue": {"type": "string"}},
            "required": ["fn", "issue"], "additionalProperties": False}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["overall_assessment", "most_serious_issues", "next_steps"],
    "additionalProperties": False,
}

SUMMARY_INSTRUCTIONS = """You write the summary of a citation-checking report for a law student or a supervising lawyer.
Use only the results provided. In 3-5 sentences, give an overall assessment of the citation accuracy of the draft.
Then list the most serious issues FIRST, in this order of seriousness: quotation not in the source / unsupported,
misattributed, overstated, wrong pinpoint, could not be checked. Be precise and use footnote numbers.
Note where the tool could not verify something, without guessing whether the citation is right."""


def summarise_report(results: list, effort: str = "low") -> dict:
    lines = [f"fn {r['unit_id']} | {r['citation_text']} | verdict {r['verdict']} | rule {r['rule']} | "
             f"{r['explanation']} | flags: {'; '.join(r['flags']) or 'none'}" for r in results]
    return ask_ai("Results:\n" + "\n".join(lines), instructions=SUMMARY_INSTRUCTIONS, effort=effort,
                  schema=SUMMARY_SCHEMA, step="11 summarise report")


def sort_by_severity(results: list) -> list:
    return sorted(results, key=lambda r: (-SEVERITY.get(r["verdict"], 0), r["fn"]))


def pinpoint_summary(r: dict) -> str:
    status = (r.get("pinpoint_check") or {}).get("status")
    if r["verdict"] == "WRONG_PINPOINT":
        return "✗ wrong"
    return {"consistent": "✓", "inconsistent": "✗", "unverifiable": "unverifiable",
            "no_pinpoint": "none given", "not_checked": "not checked"}.get(status, "—")


def results_rows(results: list) -> list:
    """Flat rows for the table and the CSV."""
    rows = []
    for r in results:
        j = r.get("judgement") or {}
        fix = r.get("fix") or {}
        rows.append({
            "fn": r["unit_id"],
            "proposition": r["proposition"],
            "citation": r["citation_text"],
            "matched source": r["source_label"],
            "verdict": r["verdict"],
            "pinpoint": pinpoint_summary(r),
            "confidence": round(r["confidence"], 2) if r.get("confidence") is not None else None,
            "who is speaking": j.get("who_is_speaking", ""),
            "found at": j.get("found_at_para"),
            "supporting quote": j.get("supporting_quote", ""),
            "rule applied": r["rule"],
            "explanation": r["explanation"],
            "flags": " | ".join(r["flags"]),
            "suggested footnote": fix.get("suggested_footnote", ""),
            "suggested sentence": fix.get("suggested_sentence", ""),
            "fix explanation": fix.get("explanation", ""),
            "turns": len(r["turns"]),
        })
    return rows


def to_csv(results: list) -> bytes:
    rows = results_rows(results)
    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")   # BOM so Excel shows curly quotes correctly


def to_docx(run: dict) -> bytes:
    """A Word report: summary, results by severity, settings and the AI-call log."""
    doc = Document()
    doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading("Footnote–Source Verifier report", 0)
    doc.add_paragraph(f"Draft: {run['draft_name']}   ·   Generated {datetime.now():%d %B %Y, %H:%M}")
    doc.add_paragraph(f"AI calls this run: {run['ai_call_count']}   ·   embedding requests: "
                      f"{run['embedding_requests']}   ·   runtime: {run['runtime_seconds']:.0f} s")

    summary = run.get("summary") or {}
    doc.add_heading("Summary", 1)
    doc.add_paragraph(summary.get("overall_assessment", ""))
    for issue in summary.get("most_serious_issues", []):
        doc.add_paragraph(f"fn {issue['fn']}: {issue['issue']}", style="List Bullet")

    doc.add_heading("Results (most serious first)", 1)
    for r in sort_by_severity(run["results"]):
        heading = doc.add_heading(f"fn {r['unit_id']} — {r['verdict']}", 2)
        if r["verdict"] != "SUPPORTED":
            heading.runs[0].font.color.rgb = RGBColor(0xA0, 0x10, 0x10)
        j = r.get("judgement") or {}
        for label, value in [
            ("Proposition", r["proposition"]), ("Citation", r["citation_text"]),
            ("Matched source", r["source_label"]), ("Rule applied", r["rule"]), ("Why", r["explanation"]),
            ("Who is speaking", f"{j.get('who_is_speaking', '')} {j.get('speaker_detail', '')}".strip()),
            ("Supporting quote", j.get("supporting_quote", "")), ("Pinpoint", pinpoint_summary(r)),
            ("Flags", "; ".join(r["flags"])),
            ("Suggested footnote", (r.get("fix") or {}).get("suggested_footnote", "")),
            ("Suggested sentence", (r.get("fix") or {}).get("suggested_sentence", "")),
            ("Fix", (r.get("fix") or {}).get("explanation", "")),
        ]:
            if value:
                p = doc.add_paragraph()
                p.add_run(f"{label}: ").bold = True
                p.add_run(str(value))

    doc.add_heading("Verdict definitions", 1)
    for name, meaning in VERDICTS.items():
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(name + ": ").bold = True
        p.add_run(meaning)

    doc.add_heading("Run settings", 1)
    for key, value in run["settings"].items():
        doc.add_paragraph(f"{key}: {value}", style="List Bullet")

    doc.add_heading("AI-call log", 1)
    table = doc.add_table(rows=1, cols=5)
    table.style = "Light Grid Accent 1"
    for cell, title in zip(table.rows[0].cells, ["#", "step", "model", "seconds", "tokens in/out"]):
        cell.text = title
    for call in run["call_log"]:
        cells = table.add_row().cells
        cells[0].text = str(call["n"] or "failed")
        cells[1].text = call["step"] + (f" ({call['error']})" if call["error"] else "")
        cells[2].text = call["model"]
        cells[3].text = str(call["seconds"])
        cells[4].text = f"{call['input_tokens'] or ''}/{call['output_tokens'] or ''}"

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
