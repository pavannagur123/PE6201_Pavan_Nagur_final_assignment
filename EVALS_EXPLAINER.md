# MacroFit Evaluation Explainer

## Evaluation question

Python can guarantee budget and numeric constraints, so constraint accuracy alone cannot justify the language model. The evaluation therefore separates two questions:

1. Does the complete pipeline take the expected action for normal, ambiguous, infeasible, unsafe, and hostile inputs?
2. When AI Layer 2 changes the optimizer’s valid selection, does a blind human evaluator prefer the AI plan?

## Twenty orchestrated cases

The authoritative notebook contains 20 manually written cases. Twelve are normal meal requests covering cuisine, spice, portability, heaviness, dietary choices, meal-specific requirements, and variety. Eight are outliers covering a same-meal dietary contradiction, a S$2 budget, unclear meal references, two prompt attacks, a complex valid preference, and an extreme conflict between refusing food and requesting a very high intake.

Each case declares one expected pipeline outcome: `layer_2`, `clarification`, `infeasible`, `guardrail_blocked`, `safety_stop`, or `technical_error`. Evaluation mode is non-interactive. When clarification is required, the case records that result and moves to the next case. The separate live workflow asks the user up to two follow-up questions.

## Stages recorded

Each result retains the deterministic precheck, Layer 1 JSON, Python result, Layer 2 JSON, optimizer plan, AI plan, error category, and execution mode where applicable. A stage that should not run remains empty. This makes early stopping visible rather than treating it as missing output.

Layer 1 interprets flexible language into typed hard and soft preferences. Python then calculates feasibility and creates a diverse shortlist. Layer 2 may select only an exact plan ID from that shortlist. Python validates the returned ID before display.

## Latest result

The latest run passed 19 of 20 cases, or 95%. It made 13 Layer 1 calls and 12 Layer 2 calls. The two prompt-injection cases were blocked before a model call. Clarification and infeasibility cases also stopped at the intended stage.

The failed case was E06: “Spicy non-veg lunch, light vegetarian dinner.” The precheck incorrectly matched `veg` inside `non-veg` and reported a lunch contradiction. The intended correction is to match standalone `veg` only when it is not preceded by `non-`. This is a deterministic parser defect and should be rerun after correction.

## Blind comparison

The optimizer and AI selected different plan IDs in 7 of the 12 Layer 2 cases. Five selections were identical. Identical selections should be reported as `same_selection`; they are not meaningful blind comparisons.

Five blind judgements have been entered, all as ties. This does not yet demonstrate that Layer 2 improves what a person would choose to eat. It may mean the compared plans were similarly attractive, the judgement sample was too small, or some recorded ties were actually identical selections. The remaining changed-plan cases should be judged using randomly labelled A and B plans without revealing their source.

The model preference win rate is:

```text
AI wins / (AI wins + optimizer wins)
```

Ties and same selections are reported separately. The 20-case exercise is directional evidence, not a population study.

## Cost measurement

The recorded evaluation used 25 API calls, 25,997 prompt tokens, and 2,606 completion tokens. Provider-reported evaluation cost was US$0.00546315. The reported US$0.00109263 figure is best labelled cost per valid judged recommendation. Cost per AI-preferred win is currently unavailable because no AI win has been observed.

## Interpreting the result

The 95% pass rate supports the pipeline design and the outlier coverage. It does not prove model value. The strongest evidence for Layer 2 will come from the seven cases where it changed the choice and a blind evaluator expressed a preference. Reporting this distinction prevents optimizer success from being credited to the model.

## Reproduction

In `MacroFit_AI_Orchestrated_COMPLETE.ipynb`, set `CASE_IDS_TO_RUN = None`, run the evaluation cells, enter blind judgements only for different selections, and export the summary and detailed JSON. Preserve model name, token counts, provider cost, prompts, schemas, dataset version, and evaluation date with the result.
