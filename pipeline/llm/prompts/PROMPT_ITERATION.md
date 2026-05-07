# Prompt Iteration Log

Prompts live in `pipeline/llm/prompts/`. The current production prompt is
`latest.txt`. Historical versions are preserved as `c1v1prompt.txt` etc. for
reproducibility — benchmark runs reference the specific version used.

---

## Cycle 1

### c1v1prompt.txt
Initial prompt. Binary + uncertain classification. Minimal instructions.
No context support. Schema: `classification`, `reasoning`, `recommendation`.

### c1v2prompt.txt
Added `[v2]` and `V2:` markers to `reasoning` and `recommendation` fields.
Purpose: distinguish c1v2 outputs from c1v1 outputs in CSV logs without
parsing the full response text. No functional change to task instructions.

---

## Cycle 2

### c2v1prompt.txt
Dropped `uncertain` from classification — binary only (`fire_detected` /
`no_fire_detected`). Still identical task framing to c1v2 otherwise.
Kept `[v2]` / `V2:` markers (carry-over, later dropped).

### c2v2prompt.txt
Added `situation_brief` and `tactical_priority` fields to schema.
Both nullable when no operational context is provided.
First version to acknowledge the GIS context block in instructions.

### c2v3prompt.txt
Major rewrite. Introduced full role framing ("analyst supporting wildfire
response"). Explicit input framing — image tells you what, context tells you
where. Image-context scale mismatch addressed ("do not attempt to identify
context features in the image"). Added constraint against resource
quantities, equipment types, and timeline estimates.
Removed `[v2]` / `V2:` markers.

### c2v4prompt.txt
Added source attribution instruction: "From the image..." / "From the
context...". Added explicit wind reasoning instruction — use wind direction
when present, do not infer when absent. Added constraint against directional
fire spread assumptions without wind data (fix for model asserting specific
flanks without evidence).

### c2v5prompt.txt
Added image primacy clause to classification field: classification must be
based solely on visual evidence; context must not influence fire/no-fire
determination. Fix for context bias false positives on fog/low-visibility
images where operational framing caused the model to classify no-fire scenes
as fire.

### latest.txt ← current production prompt
Added tactical priority decision criteria with explicit thresholds:
- suppress / contain / evacuate / monitor definitions
- Explicit rule: when permanent structures or settlements are within 1 km,
  prioritise evacuate or contain over suppress.
Fix for model defaulting to suppress in all scenarios regardless of asset
proximity (observed in scenario eval: scenario 02 with 83 buildings and
Tyresta by 666m away still returned suppress).

---

## Key decisions

| Decision | Version | Rationale |
|---|---|---|
| Drop `uncertain` | c2v1 | Forces binary commit; ambiguous cases belong in Limitations |
| Add situation_brief + tactical_priority | c2v2 | Cycle 2 schema requirement; radio-ready output |
| Image-context scale framing | c2v3 | Prevents model matching context features to image objects |
| Source attribution | c2v4 | Makes evidence chain explicit; surfaces hallucinated terrain claims |
| Wind reasoning | c2v4 | Model was inferring spread direction without wind data |
| Image primacy clause | c2v5 | Context bias caused false positives on fog images |
| Tactical priority criteria | latest | Model defaulted to suppress; explicit thresholds fix evacuation cases |