"""
RESULTS PAGE - summary banner, colour-coded table, one expander per citation
unit, downloads and the AI-call log. Everything is read from the saved run in
st.session_state, so it survives reruns (clicking a download, opening an
expander) without re-running the pipeline.
"""

import pandas as pd
import streamlit as st

from reusable_code.grounding import format_paras
from reusable_code.report import pinpoint_summary, results_rows, sort_by_severity, to_csv, to_docx
from reusable_code.verdict_rules import VERDICT_COLOURS, VERDICT_NAMES, VERDICTS

ICONS = {"SUPPORTED": "✅", "OVERSTATED": "🟡", "MISATTRIBUTED": "🔴", "WRONG_PINPOINT": "🟠",
         "UNSUPPORTED": "⛔", "CANNOT_DETERMINE": "⚪"}


def _banner(run: dict):
    results = run["results"]
    counts = {v: sum(r["verdict"] == v for r in results) for v in VERDICT_NAMES}
    st.subheader("Results")
    top = st.columns(4)
    top[0].metric("AI calls this run", run["ai_call_count"],
                  help="Calls to the reasoning model (the brief requires at least 5). Embedding requests are counted separately.")
    top[1].metric("Citation units checked", len(results),
                  help=f"From {len(set(r['fn'] for r in results))} footnote(s); a footnote citing two sources is two units.")
    top[2].metric("Embedding requests", run["embedding_requests"])
    top[3].metric("Runtime", f"{run['runtime_seconds']:.0f} s")
    cols = st.columns(len(VERDICT_NAMES))
    for col, name in zip(cols, VERDICT_NAMES):
        col.metric(f"{ICONS[name]} {name.replace('_', ' ').title()}", counts[name])


def _summary(run: dict):
    summary = run.get("summary") or {}
    st.info(summary.get("overall_assessment", ""))
    if summary.get("most_serious_issues"):
        st.markdown("**Most serious issues first**")
        for issue in summary["most_serious_issues"]:
            st.markdown(f"- **fn {issue['fn']}**: {issue['issue']}")
    if summary.get("next_steps"):
        with st.expander("Suggested next steps"):
            for step in summary["next_steps"]:
                st.markdown(f"- {step}")


def _sources_table(run: dict):
    with st.expander("Sources as identified (step 3) - check these are right", expanded=False):
        rows = [{"file": s["filename"], "identified as": s["label"],
                 "court": s["identity"].get("court", ""), "date": s["identity"].get("decision_date", ""),
                 "reported citations printed": ", ".join(s["identity"].get("reported_citations", [])),
                 "paragraphs": s["n_paragraphs"], "numbering": s["numbering_style"],
                 "readable": "yes" if s["readable"] else "NO (scanned)"} for s in run["sources"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _table(run: dict):
    columns = ["fn", "verdict", "pinpoint", "confidence", "citation", "proposition"]
    frame = pd.DataFrame(results_rows(run["results"]))[columns]
    styled = frame.style.map(lambda v: f"background-color: {VERDICT_COLOURS.get(v, '')}", subset=["verdict"])
    st.dataframe(styled, hide_index=True, width="stretch",
                 column_config={"fn": st.column_config.TextColumn(width="small"),
                                "pinpoint": st.column_config.TextColumn(width="small"),
                                "confidence": st.column_config.NumberColumn(format="%.2f", width="small"),
                                "citation": st.column_config.TextColumn(width="medium"),
                                "proposition": st.column_config.TextColumn(width="large")})


def _unit_expander(r: dict):
    j = r.get("judgement") or {}
    title = f"{ICONS[r['verdict']]} fn {r['unit_id']} — {r['verdict']} — {r['citation_text'][:80]}"
    with st.expander(title):
        st.markdown(f"**Why:** {r['explanation']}  \n*Rule applied: {r['rule']}*")
        for flag in r["flags"]:
            st.warning(flag)

        st.markdown(f"**Draft sentence:** {r['sentence']}")
        st.markdown(f"**Proposition checked:** {r['proposition']}  \n**Draft attributes it to:** {r['attributed_to']}")
        if r.get("authorial_comment"):
            st.caption(f"Author's own comment (not checked against the source): {r['authorial_comment']}")
        st.markdown(f"**Citation:** {r['citation_text']}  \n**Resolved to:** {r['full_citation']}  \n"
                    f"**Matched source:** {r['source_label']} — {r['match_reason']}")
        for note in r["resolution_notes"]:
            st.caption("↳ " + note)

        if r["quote_results"]:
            st.markdown("**Grounding check (i): quoted phrases in the cited source**")
            for q in r["quote_results"]:
                if q["found"]:
                    st.markdown(f"- ✓ ‘{q['phrase']}’ found at {format_paras(q['paras'][:8])}")
                else:
                    elsewhere = "; ".join(f"{o['label']} {format_paras(o['paras'])}" for o in q["found_elsewhere"])
                    st.markdown(f"- ✗ ‘{q['phrase']}’ **not found** in the cited source"
                                + (f" — but found in **{elsewhere}**" if elsewhere else ""))

        if j:
            st.markdown("**AI element findings (step 7)**")
            st.markdown(f"- Content: `{j['content_match']}` · Attribution: `{j['attribution_match']}` · "
                        f"Strength: `{j['strength_match']}` · Found at: [{j['found_at_para']}] · "
                        f"Confidence: {j['confidence']:.2f} · AI's own verdict: `{j['verdict']}`")
            st.markdown(f"- **Who is speaking:** {j['who_is_speaking'].replace('_', ' ')} — {j['speaker_detail']}")
            st.markdown(f"- **Supporting quote:** “{j['supporting_quote']}”")
            st.markdown(f"- **Reasoning:** {j['reasoning']}")
        if r.get("support_check"):
            mark = "✓" if r["support_check"]["found"] else "✗"
            st.markdown(f"**Grounding check (ii):** {mark} {r['support_check']['detail']}")
        if r.get("pinpoint_check"):
            st.markdown(f"**Grounding check (iii), pinpoint:** {pinpoint_summary(r)} — {r['pinpoint_check'].get('detail', '')}")

        if r.get("fix"):
            fix = r["fix"]
            st.markdown(f"**Suggested fix ({fix['fix_type'].replace('_', ' ')}):** {fix['explanation']}")
            if fix.get("suggested_sentence"):
                st.markdown(f"> {fix['suggested_sentence']}")
            if fix.get("suggested_footnote"):
                st.code(fix["suggested_footnote"], language=None)

        if r["evidence"]:
            with st.popover(f"Evidence passages ({len(r['evidence'])})"):
                for e in r["evidence"]:
                    meta = e["origin"] + (f" · {e['note']}" if e["note"] else "") + \
                        (f" · distance {e['distance']}" if e["distance"] is not None else "")
                    st.markdown(f"**[{e['para']}]** _{meta}_")
                    st.text(e["text"])
        if r["turns"]:
            st.markdown("**Reflection work record (step 9)**")
            st.dataframe(pd.DataFrame([{k: t[k] for k in ("turn", "queries", "evidence", "ai_verdict", "verdict",
                                                            "confidence", "supporting_quote_found", "stop_reason")}
                                       for t in r["turns"]]), hide_index=True, width="stretch")


def show_results(run: dict):
    _banner(run)
    _summary(run)
    _sources_table(run)
    st.markdown("#### Citation table")
    _table(run)

    st.markdown("#### Details (most serious first)")
    for r in sort_by_severity(run["results"]):
        _unit_expander(r)

    st.markdown("#### Downloads")
    c1, c2 = st.columns(2)
    stem = run["draft_name"].rsplit(".", 1)[0]
    c1.download_button("⬇️ Results (CSV)", to_csv(run["results"]), file_name=f"{stem}_citation_check.csv",
                       mime="text/csv", on_click="ignore", width="stretch")
    c2.download_button("⬇️ Report (.docx)", to_docx(run), file_name=f"{stem}_citation_report.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       on_click="ignore", width="stretch")

    with st.expander(f"AI-call log ({run['ai_call_count']} calls)"):
        st.dataframe(pd.DataFrame(run["call_log"]), hide_index=True, width="stretch")
    with st.expander("Verdict definitions"):
        for name, meaning in VERDICTS.items():
            st.markdown(f"- {ICONS[name]} **{name}**: {meaning}")
