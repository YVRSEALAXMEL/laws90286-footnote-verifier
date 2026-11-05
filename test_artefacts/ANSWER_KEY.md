# Answer key: ground truth for the three test cases

Each claim was checked by hand against the source PDFs in `sources/`. "Ground truth" is the correct verdict. "Likely tool behaviour" is the prediction to test against (and to write about in Assignment 3).

Verdict set: SUPPORTED · OVERSTATED · MISATTRIBUTED · WRONG_PINPOINT · UNSUPPORTED · CANNOT_DETERMINE

## Sources
| File | Case | Format notes |
|---|---|---|
| `Brown_v_The_Queen_2020_VSCA_212.pdf` | *Brown v The Queen* [2020] VSCA 212; reported as (2020) 62 VR 491 | BarNet Jade print, 59 pp. Paragraphs are numbered "1.", "2." etc. **Jade inserts "Following paragraph cited by:" blocks after many paragraphs.** These quote *later* judgments (e.g. *DPP (Cth) v Heaton*, *DPP v Herrmann*, *Gordon v Tasmania*), each with its own paragraph numbers (e.g. "139.", "59."). The case header shows only the medium-neutral citation; "62 VR 491" appears only inside the cited-by blocks. No VR page numbers anywhere. User email redacted. |
| `Kapua_v_R_2023_NSWCCA_14.pdf` | *Kapua v R* [2023] NSWCCA 14 | BarNet Jade print, 36 pp. Paragraphs numbered "111." etc. No cited-by blocks. User email redacted. |
| `R_v_Lojovic_2025_ONCA_319.pdf` | *R v Lojovic* [2025] ONCA 319 | CanLII, 22 pp. Paragraphs numbered "[42]" etc. "Page: N" headers and "2025 ONCA 319 (CanLII)" watermark lines. |

## TC1: `TC1_accurate_citations.docx` (real extract from the user's paper). All claims should pass.
| fn | Claim (short) | Citation | Ground truth | Evidence |
|---|---|---|---|---|
| 1 | Four arsons; house destroyed; 'strong causal links'; judge bound by O'Neill; 'materially reduced the sentence' | Brown (2020) 62 VR 491, [1]–[3] | SUPPORTED | [1] four charges, house destroyed; [2] 'strong causal links to the offending'; [3] 'materially reduced the sentence to be imposed' |
| 2 | 'no different position from any other offender…'; 'should no longer be followed' | Ibid [6], [29] | SUPPORTED | [6] |
| 3 | Director's concession 'whether of such disorders in general…could not be sustained' | Ibid [28]–[29] | SUPPORTED | [28] |
| 4 | 'rigorously scrutinised'; 'some severity'; 'no more than descriptive of maladaptive behaviour' | Brown (n 1) [6], [56]–[58], [68]–[69] | SUPPORTED | [6], [68], [69] |
| 5 | 'No such requirement operated.' | Kapua [2023] NSWCCA 14, [111] | SUPPORTED | [111] |
| 6 | Ontario trial judge: 'worse', 'total' | Lojovic [2025] ONCA 319, [42]–[44] | SUPPORTED | [43] 'worse', [44] 'total' |
| 7 | 'a degree of planning, system, and persistence' | Kapua (n 5) [113] | SUPPORTED | [113] |

What this test shows: footnote extraction from .docx; resolving Ibid and (n X) references; matching a parallel citation (62 VR 491) to a file labelled with the medium-neutral citation; ranges and multiple pinpoints; three different paragraph-numbering formats.

## TC2: `TC2_seeded_errors.docx`. Five planted errors and one control.
| fn | Claim (short) | Citation | Ground truth | Why |
|---|---|---|---|---|
| 1 | The **Court of Appeal** held ability to anticipate episodes made offence 'worse', responsibility 'total' | Lojovic [42]–[44] | MISATTRIBUTED | At [42]–[44] the Court of Appeal is describing the **trial judge's** reasons. At [52]–[54] it held that treating the disorder as aggravating was "an error in principle". The quoted words are present, so a simple quote check passes; only reasoning about who said it catches the error. |
| 2 | A condition can **never** reduce culpability where offending involves planning | Kapua [112]–[113] | OVERSTATED | [112]: a condition "may, or may not, have an effect upon sentence"; [113]: planning was "not irrelevant" to the assessment. |
| 3 | 'No such requirement operated.' | Kapua (n 2) [113] | WRONG_PINPOINT | Proposition and quote correct, but at [111], not [113]. |
| 4 | Brown held a personality disorder will 'ordinarily operate to reduce moral culpability' | Brown [6] | UNSUPPORTED | Fabricated quotation: the phrase appears nowhere in Brown. The quote-grounding check should fail. |
| 5 | In Brown the sentencing judge observed 'a degree of planning, system, and persistence' | Brown (n 4) [35] | UNSUPPORTED (misattributed to the wrong case) | The words are from Kapua [113], another uploaded source. Tests whether retrieval stays inside the cited source. |
| 6 | 'diagnosis is not the key. Impairment is the key' | Ibid [52] | SUPPORTED | Brown [52]. A control, to check that the tool doesn't flag everything. |

## TC3: `TC3_limitations.docx`. Designed to expose limitations.
| fn | Claim (short) | Citation | Ground truth | Likely tool behaviour / limitation shown |
|---|---|---|---|---|
| 1 | Brown described the Verdins principles as 'exceptional', to be established on the balance of probabilities | Brown [6] | UNSUPPORTED | The words appear in the PDF, but only in Jade's "cited by" blocks after Brown [6], quoting *later* judgments. A paragraph-based splitter files them under Brown [6], so retrieval, the verdict **and** the deterministic quote check are all likely to pass a false claim. This shows that the tool trusts the source file as given. |
| 2 | Verdins restated the principle; culpability 'as distinct from legal responsibility' | R v Verdins (2007) 16 VR 269, [32] | CANNOT_DETERMINE (source not uploaded; the claim is in fact accurate) | Risks: (a) the model "verifies" it from its own memory; (b) the matching step wrongly uses Brown, which discusses Verdins at length. The correct behaviour is to refuse to verify. |
| 3 | Defence expert: 'strong causal links to the offending' | Brown (n 1) 493 | SUPPORTED proposition; pinpoint CANNOT be verified | The pinpoint is a **page of the 62 VR report**, but the uploaded copy is the medium-neutral version with paragraph numbers only (the quote is at [2]). The tool should say the pinpoint is unverifiable, not that it's wrong. |
| 4 | DB 'lacked the internal capacity…'; 'fundamentally impaired her ability…'; 'language close to Strawson's invitation…' | Brown (n 1) [81]–[82]; Strawson (1962) 187, 193–4 | Brown part SUPPORTED ([81], [82]); Strawson part CANNOT_DETERMINE; "language close to" is the author's own argument, not a cited proposition | Tests splitting a footnote with two sources, partial verification, and telling argument apart from attribution. The model may mark the whole footnote UNSUPPORTED or SUPPORTED. |
