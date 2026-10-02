# MacroFit Evaluation Explainer

## Evaluation question

Python can guarantee budget and numeric constraints, so constraint accuracy alone cannot justify the language model. The evaluation therefore separates two questions:

An infeasible case remains an infeasible evaluation outcome even when Python supplies a separately labelled closest numeric fallback. The fallback is not included in the feasible shortlist, is not ranked by Layer 2, lists its exact deviations, and requires user confirmation.

The evaluator also derives explicit requirements directly from each original prompt and compares them with the final selected plan metadata. Dietary type, meal-specific spice, heaviness, portability, avoided formats, required cuisine/format and repetition are checked. A routing success cannot hide a recommendation contradiction. The live GPT-4o-mini run records 18/20 correct outcomes (90%), 25 API calls, 26,521 tokens, US$0.00525165 cost, no technical errors and no final-plan request violations. It uses a disclosed 30 mg (1%) tolerance around the 3,000 mg sodium policy ceiling because the synthetic estimates are rounded; actual sodium remains visible.

The raw Layer 2 rationale is retained for critique, but the application displays a Python-generated explanation built from trusted plan metadata. This prevents a fluent but inaccurate model sentence from becoming a silent user-facing failure.

1. Does the complete pipeline take the expected action for normal, ambiguous, infeasible, unsafe, and hostile inputs?
2. When AI Layer 2 changes the optimizer’s valid selection, does a blind human evaluator prefer the AI plan?

## Twenty orchestrated cases

The authoritative notebook contains 20 manually written cases. They cover cuisine, portability, heaviness, variety, explicit meal-specific spice, `non-veg`, meal-boundary wording, negation, and an `anything` reset. Outliers cover same-meal contradiction, a S$2 budget, unclear references, prompt attacks, and an extreme conflict between refusing food and requesting a very high intake.

Each case declares one expected pipeline outcome: `layer_2`, `clarification`, `infeasible`, `guardrail_blocked`, `safety_stop`, or `technical_error`. Evaluation mode is non-interactive. When clarification is required, the case records that result and moves to the next case. The separate live workflow asks the user up to two follow-up questions.

## Stages recorded

Each result retains the deterministic precheck, Layer 1 JSON, Python result, Layer 2 JSON, optimizer plan, AI plan, error category, and execution mode where applicable. A stage that should not run remains empty. This makes early stopping visible rather than treating it as missing output.

Layer 1 interprets flexible language into typed hard, meal-spice, and soft preferences. Python then calculates feasibility and creates at most three plans. Plans are unique by item set, so lunch/dinner permutations of the same dishes cannot inflate diversity. Layer 2 sees actual food metadata and may select only an exact plan ID. Python validates the returned ID before display.

## Reworked version status

The earlier 19/20 result belongs to the superseded pipeline and must not be presented as the result of this reworked version. E06's `non-veg` substring defect is fixed. E10–E12 now assert the exact Layer 1 semantics for `chicken for lunch, veg dinner`, `no chicken for dinner`, and `veg lunch, anything for dinner`. Meal-specific spicy cases also assert the resulting dinner or lunch metadata rather than accepting any successful model call.

Run all 20 cases again before submission and use only the newly exported result files. A changed pipeline invalidates the earlier pass rate and cost evidence.

## Blind comparison

Only newly generated cases where optimizer and AI choose different item-set-unique plans are eligible for blind comparison. Identical selections are recorded as `same_selection`; they are not blind evidence. Judge randomly labelled A and B without revealing their source.

The model preference win rate is:

```text
AI wins / (AI wins + optimizer wins)
```

Ties and same selections are reported separately. The 20-case exercise is directional evidence, not a population study.

## Cost and token measurement

The notebook reads prompt tokens, completion tokens, and per-call cost from OpenRouter's response `usage` object. If per-call cost is absent, it uses the before/after key-usage delta. The local `token_proxy` exists only to cap payload size and must never be cited as provider token usage. Report cost per judged recommendation only after a human choice exists; report cost per AI-preferred win only when at least one AI win exists.

## Interpreting the result

Pipeline pass rate and model preference are separate. Python owns constraint success. Model value comes only from changed selections that a blind evaluator prefers. Reporting this distinction prevents optimizer success from being credited to AI.

The live evaluation produced 18 correct outcomes from 20 cases, giving a 90% pass rate. The other two cases failed openly instead of returning unsuitable food. E05 asked for a hearty vegetarian dinner, but the synthetic menu had no matching item. E07 asked for portable, non-soupy food for the whole day, but no combination could meet both the budget and nutrition limits. No selected final plan broke the hard constraints checked by Python.

Manual review did find five problems in the raw Layer 2 explanations. The meals themselves were valid, but some explanations discussed preferences the user had not requested or made claims the data could not support. One response criticised a meal for not being spicy even though the user had not asked for that meal to be spicy. Another claimed to satisfy “warm comfort food,” although warmth is not recorded in the dataset. These were explanation problems, not incorrect meal selections. To keep them away from the user, the notebook displays an explanation built from verified menu fields. The raw model wording remains in the detailed JSON for inspection.

The checks cover the qualities the dataset actually records: dietary type, meal-specific spice, heaviness, portability, avoided formats, cuisine, food format, and repetition. They cannot prove personal ideas such as warmth, comfort, or familiarity. A pass therefore means that the implemented requirements were met; it does not mean that every subjective phrase was objectively verified.

## Reproduction

In `MacroFit_AI_Orchestrated_COMPLETE.ipynb`, set `CASE_IDS_TO_RUN = None`, run the evaluation cells, enter blind judgements only for different selections, and export the summary and detailed JSON. Preserve model name, token counts, provider cost, prompts, schemas, dataset version, and evaluation date with the result.
