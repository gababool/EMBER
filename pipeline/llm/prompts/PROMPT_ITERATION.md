# Prompt Iteration Log

Prompts live in `pipeline/llm/prompts/`. The current production prompt is
`latest.txt`. Historical versions are preserved as `c1v1prompt.txt` etc. for
reproducibility — benchmark runs reference the specific version used.

The system role content is stored separately in `system_content.txt` and loaded
at runtime by both the pipeline and benchmark inference modules. Historical
versions are preserved as `system_content_v1.txt` etc.

---

## System Content

### system_content_v1.txt
Original hardcoded system role content. Minimal — role and JSON format only.
```
You are a wildfire analyst. You must respond strictly in JSON format.
```

### system_content_v2.txt
Extended version with detailed role framing, operational context, and
decision-support boundary. Caused Qwen3-VL 4B context window exhaustion
on complex images — thinking mode consumed token budget before producing output.

### system_content.txt ← current (v3)
Minimal but improved over v1. Adds early-response phase context and explicit
human authority boundary without repeating content covered in the main prompt.
```
You are a wildfire early-response decision-support analyst. Humans retain
final decision authority. Respond strictly in JSON format.
```

Rationale for keeping it short: v2 caused Qwen3-VL failures. The main prompt
already covers inputs, task framing, and reasoning instructions — the system
role should only carry what is not said elsewhere.

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

### c2v6prompt.txt
One change from c2v5:

1. **Tactical priority decision criteria added**: explicit suppress / contain /
   evacuate / monitor definitions with a 1 km settlement threshold rule. Fix
   for models defaulting to suppress regardless of asset proximity (observed
   in scenario eval: scenario 02 with 83 buildings and Tyresta by 666m away
   still returned suppress).

### latest.txt (c2v7) ← current production prompt
Two changes from c2v6:

1. **Tactical priority removed**: field too volatile across minor prompt
   iterations to be a reliable signal. Observed to flip between values with
   small wording differences that should not affect strategic priority.
   Removed from schema entirely in both pipeline and benchmark.

2. **Recommendation changed to bullet points**: 3-5 action-oriented bullets
   each starting with a verb, referencing named context features. Following
   supervisor feedback on actionable structured output. Ministral follows this
   well; Qwen3-VL produces thinking mode leakage into the recommendation field
   when generating structured lists — documented as a model limitation, not
   addressed in the prompt.

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
| Tactical priority criteria | c2v6 | Model defaulted to suppress; explicit thresholds fix evacuation cases |
| Drop tactical_priority | c2v7 | Too volatile across minor prompt iterations to be a reliable signal |
| Bullet point recommendations | c2v7 | Supervisor feedback; Ministral follows well, Qwen3-VL leaks thinking mode into output fields |
| System prompt to user role | c2v7 | Fixes Qwen3-VL context exhaustion on complex image + long prompt combinations. However, Qwen can still fail through reasoning leaks. |
| Modularised system content | c2v7 | Separates role definition from task instructions; allows per-deployment tuning without altering versioned prompt file |