"""How it works: the pipeline, the verdict logic and the known limitations."""

import streamlit as st

from reusable_code.api_key import api_key_module
from reusable_code.password_gate import require_password
from reusable_code.chunking import MAX_CHUNK_TOKENS
from reusable_code.verdict_rules import CONFIDENCE_THRESHOLD, VERDICTS, WHO_IS_SPEAKING

st.set_page_config(page_title="How it works", page_icon="🧭", layout="wide")
require_password()
api_key_module()

st.title("🧭 How it works")
st.write("The tool reproduces the check a careful editor or supervising lawyer makes of each footnote: *does the "
         "cited source, at the cited pinpoint, say what the text says it says - with the same strength, and in the "
         "voice of the person the text says said it?* Work the computer can do exactly (finding quotations, "
         "reading pinpoints, resolving Ibid) is done in code; judgement (who is speaking, how strong a statement is) "
         "is done by the AI, and the AI's work is then checked by code and by a second AI reviewer.")

st.header("Pipeline")
st.graphviz_chart("""
digraph G {
  rankdir=TB; nodesep=0.35; ranksep=0.35; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
  subgraph cluster_in { label="Inputs"; style=dashed; docx [label="Draft .docx\\n(Word footnotes)", fillcolor="#eef"]; pdfs [label="Source PDFs", fillcolor="#eef"]; }
  s1 [label="1 Ingest draft\\nfootnotes + quotes\\n(code)", fillcolor="#e8f5e9"];
  s2 [label="2 Ingest sources\\nnumbered paragraphs,\\nsemchunk >400 tokens\\n(code)", fillcolor="#e8f5e9"];
  db [label="Chroma (in memory)\\ntext-embedding-3-large", shape=cylinder, fillcolor="#fff8e1"];
  s3 [label="3 Identify source\\n(AI × each PDF)", fillcolor="#e3f2fd"];
  s4 [label="4 Extract claims\\n(AI × 1)", fillcolor="#e3f2fd"];
  s5 [label="5 Resolve citations\\nIbid, (n X), parallel cites\\n(AI × 1 + code rules)", fillcolor="#e3f2fd"];
  subgraph cluster_loop { label="For each citation unit: reflection loop (hard stop after N turns)"; style=rounded; color="#999";
    s6 [label="6 Retrieve evidence\\npinpoint + quote match\\n+ AI-planned search", fillcolor="#e3f2fd"];
    s7 [label="7 Judge elements\\ncontent · attribution\\nstrength · location (AI)", fillcolor="#e3f2fd"];
    s8 [label="8 Grounding checks\\nquotes · supporting quote\\n· pinpoint (code)", fillcolor="#e8f5e9"];
    rules [label="Verdict precedence\\nrules R1–R6 (code)", fillcolor="#e8f5e9"];
    s9 [label="9 Reviewer\\nwell founded? (AI)", fillcolor="#e3f2fd", shape=diamond];
  }
  s10 [label="10 Suggest fix\\n(AI × flagged)", fillcolor="#e3f2fd"];
  s11 [label="11 Summary\\n(AI × 1)", fillcolor="#e3f2fd"];
  out [label="Table · CSV · .docx\\nAI-call log", fillcolor="#eef"];
  docx -> s1 -> s4; pdfs -> s2 -> db; s2 -> s3 -> s5; s4 -> s5 -> s6 -> s7 -> s8 -> rules -> s9;
  s9 -> s6 [label="no: new queries\\n+ work record", fontsize=9]; s9 -> s10 [label="yes / hard stop", fontsize=9];
  db -> s6 [style=dashed]; s10 -> s11 -> out;
}
""")
st.caption("Green = deterministic code, blue = AI calls. Where code can establish a fact, the code wins over the AI.")

st.header("Verdicts")
for name, meaning in VERDICTS.items():
    st.markdown(f"- **{name}**: {meaning}")

st.header("How a verdict is decided")
st.markdown(f"""
The AI does not simply announce a verdict. It reports **four element findings** (Lab 1 style), and code turns them
into a verdict, stopping at the first element that fails:

1. **Content** - does the source state the proposition? `none` → UNSUPPORTED; `partial` → OVERSTATED.
2. **Attribution** - who is speaking? Different from the draft's attribution → MISATTRIBUTED.
3. **Strength** - "may" vs "must", "can" vs "never", obiter vs holding. Draft stronger → OVERSTATED.
4. **Location** - where the support actually is (checked by code against the pinpoint).

Then the **precedence rules**, applied in code:

| Rule | Condition | Result |
|---|---|---|
| R1 | Source not uploaded | CANNOT_DETERMINE - the AI is never asked to judge from memory |
| R2 | Source has no text layer (scan) | CANNOT_DETERMINE - upload a searchable copy |
| R3 | A quotation in the draft is not in the cited source | UNSUPPORTED, overriding the AI; if it is in another uploaded source, say which |
| R4 | The AI's supporting quote is not in the source | reflect again; after the last turn CANNOT_DETERMINE |
| R5 | Supported, but the evidence is more than one paragraph from a paragraph pinpoint | WRONG_PINPOINT, with the correct paragraph |
| R6 | A page pinpoint (e.g. a VR page) against a paragraph-numbered copy | verdict kept; pinpoint "unverifiable for this version" |

The reviewer loop takes another turn if the reviewer is not satisfied, the supporting quote failed the check, the AI's
verdict contradicts its own elements, or its confidence is below {CONFIDENCE_THRESHOLD}.
""")

st.header("Who can be speaking in a judgment")
st.write("An appellate judgment describes the trial judge's reasons, summarises submissions, reports expert evidence "
         "and quotes other cases before it decides anything. The judge must place every passage in one of these:")
for name, meaning in WHO_IS_SPEAKING.items():
    st.markdown(f"- `{name}`: {meaning}")

st.header("Known limitations")
st.markdown(f"""
- **The tool trusts the source file as given.** BarNet Jade PDFs insert "Following paragraph cited by:" blocks that
  quote *later* cases, and the paragraph parser files that text under the Brown paragraph it follows. A claim built
  from those later cases' words can pass every check (Test Case 3, footnote 1). An experimental, default-off toggle
  strips these annotations using a layout heuristic.
- **Paragraph detection is heuristic.** It accepts only numbers that continue the running sequence, at the left
  margin. A numbered list item that happens to equal the next paragraph number could be mistaken for a paragraph.
- **Page pinpoints cannot be checked** against a medium-neutral copy (no report page numbers in the file).
- **Attribution and strength are AI judgements.** They can vary between runs, and the reviewer loop reduces but
  does not remove that. Run important checks twice.
- **Only uploaded sources are checked.** Articles, books, statutes and cases not uploaded are CANNOT_DETERMINE.
- **Scanned PDFs** are detected and refused rather than read with OCR.
- **Long paragraphs** over {MAX_CHUNK_TOKENS} tokens are split for searching; pinpoint fetches return them whole
  (up to 8,000 characters).
- **Cost and time** grow with the number of footnotes: roughly 3 AI calls per citation (plan, judge, review), more
  when the reviewer asks for another turn.
""")
