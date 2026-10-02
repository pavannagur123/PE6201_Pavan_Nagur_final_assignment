# MacroFit

MacroFit is a notebook-based meal-planning prototype for adults who depend on prepared food from restaurants, food courts, or takeaway services. It converts a profile, daily budget, and plain-language food request into a three-meal plan. The project tests a specific question: after deterministic Python has enforced the numeric and dietary constraints, does a language model select a plan that a person would rather eat?

## Persona

The primary user does not regularly cook at home, buys most meals, knows what they feel like eating, and wants to stay near a daily budget and general nutrition estimate without manually comparing many menu items. MacroFit is a general planning aid. It does not provide medical nutrition therapy, diagnosis, allergy certification, or treatment advice.

## Inputs and outputs

The live notebook accepts:

- age, weight, height, calculation sex, activity level, and goal;
- daily food budget;
- a natural-language food request;
- optional actual-intake updates later in the day.

It returns:

- estimated daily calories and macronutrients calculated by Python;
- a breakfast, lunch, and dinner shortlist that satisfies hard constraints;
- one selected plan with a plain-language explanation;
- a revised remaining-day plan after confirmed intake or substitutions;
- explicit clarification, infeasible, guardrail-blocked, safety-stop, or technical-error outcomes when a recommendation should not be produced.

## Architecture

```text
Profile and budget                 Natural-language request
        |                                   |
        v                                   v
Python target calculator       Deterministic prompt precheck
        |                                   |
        +-------------------+---------------+
                            v
                 AI Layer 1 intent extraction
                            |
                            v
           Python hard filters and plan optimizer
       budget | nutrition | meal slot | dietary rules
       no same-day duplicates | recommendation history
                            |
                            v
                 Diverse valid shortlist
                            |
                            v
              AI Layer 2 subjective plan critic
                            |
                            v
          Python validates the selected plan identifier
                            |
                            v
               Meal plan and explanation
                            |
                confirmed actual intake
                            v
             Python remaining-day recalculation
```

Python owns arithmetic, data joins, feasibility, exclusions, history, and final validation. The language model interprets flexible food language and compares subjective qualities such as spice, comfort, portability, variety, and cuisine. It cannot invent food, change prices or nutrition, relax the budget, or select a plan outside the shortlist.

## Build versus buy decision

The system decides what to build or rent at operation level. Python owns calculations and hard constraints because they must be exact, testable, and auditable. OpenRouter rents model intelligence only for language interpretation and subjective ranking. Claude was considered in the problem statement, but the final implementation is vendor-independent so the configured model can be replaced without changing the optimizer.

| Factor | Decision and reason |
|---|---|
| Cost | Rent at prototype volume; record the new provider-reported run cost before submission. Training or hosting a foundation model would be disproportionate for this prototype. |
| Latency | Keep calculations local and send compact JSON. API response time is acceptable for the notebook but has not yet been measured as a formal metric. |
| Control | Build budget, nutrition, dietary, meal-slot, history, and validation logic in Python; constrain rented model output with schemas and shortlist IDs. |
| Data gravity | Send only the user request and at most three compact synthetic plans, not the full dataset or persistent history. |
| Regulation | Keep auditable decisions local and avoid medical claims; real deployment would require privacy, residency, provider-contract, and nutrition review. |

## Run in Google Colab

1. Upload this repository folder to Google Drive.
2. Open `MacroFit_AI_Orchestrated_COMPLETE.ipynb` in Colab.
3. Add `OPENROUTER_API_KEY` to Colab Secrets and enable notebook access.
4. Confirm that `PROJECT_FOLDER` points to the uploaded folder.
5. Select **Runtime > Restart session** after replacing project files.
6. Run cells from the beginning in order.
7. Use `CASE_IDS_TO_RUN = None` for the final 20-case evaluation.

The required project files are:

```text
MacroFit_AI_Orchestrated_COMPLETE.ipynb
core.py
guardrails.py
config/
  agent_guardrails.json
  tool_descriptors.json
data/
  menu.json
  nutrition.json
  meal_rules.json
```

No web application or `app.py` is required for the submitted notebook workflow.

The configured default is `openai/gpt-4o-mini`. The alternative paid model slug `qwen/qwen3.8-flash` also exists on OpenRouter as of 2 October 2026, but evaluation results must always record the exact model actually used. The notebook and proposal use GPT-4o-mini consistently.

## Notebook flow

Run the notebook sequentially. It loads and validates the data, calculates the profile targets, collects the budget and request, performs the precheck, calls AI Layer 1, creates the Python shortlist, calls AI Layer 2 when subjective judgement is needed, displays the plan, and saves recommended item IDs in session history. The later cells accept an actual-intake message and recalculate the remainder of the day.

Live actual-intake clarification is interactive. Evaluation is non-interactive: a case requiring clarification is recorded with that outcome, and the evaluator moves to the next case.

Recommendation history persists only during the active Colab runtime. Set `RESET_HISTORY = True` for one run to clear it, then return it to `False`.

## Data

The prototype joins 60 synthetic menu rows from 10 synthetic outlets to 60 nutrition rows by `item_id`. The menu provides price and preference labels. Nutrition provides calories, protein, carbohydrates, fat, fibre, sodium, mapping method, source family, mapper, date, and confidence. Vegetarian means no meat, fish, or egg; egg dishes use a separate `egg` category.

The data validates the pipeline. It does not claim current prices or medically verified nutrition. See [DATA_EXPLAINER.md](DATA_EXPLAINER.md) for row definitions, checks, provenance, limitations, and replacement requirements for a real pilot.

## Evaluation

The notebook contains 20 manually written cases covering ordinary preferences, meal-boundary language, negation, `anything` resets, meal-specific spice, contradictions, ambiguous references, an infeasible budget, prompt attacks, and an extreme intake conflict. E06 now verifies that `non-veg` is not mistaken for `veg`; E10–E12 verify `chicken for lunch, veg dinner`, `no chicken for dinner`, and `veg lunch, anything for dinner`.

The reworked version intentionally does not reuse the previous 19/20 or cost figures because its constraints and cases changed. Run all cells with `CASE_IDS_TO_RUN = None`, then cite only the newly saved JSON and CSV output. Token counts and cost come from OpenRouter's response `usage` object or the account-usage delta. `token_proxy` is only a local payload safety cap and is never reported as provider usage.

Python sends at most three plans to Layer 2. Every plan contains the actual breakfast, lunch, and dinner names plus dietary type, spice, heaviness, format, portability, quantity, and totals. The shortlist is deduplicated by item set, so swapping the same dishes between lunch and dinner cannot create fake A/B variety.

Initial plans must stay within budget, reach the 90% calorie/protein/carbohydrate floors, remain at or below 110% of calories and 120% of fat, and stay below the prototype 3,000 mg sodium policy ceiling. The notebook allows a documented 30 mg (1%) measurement tolerance because the synthetic nutrition values are rounded; actual sodium is still displayed. Sodium above the preferred 2,000 mg level receives an optimizer penalty. This is a prototype constraint, not a health recommendation. If no plan passes every numeric limit, the result remains `available=false` and is counted as infeasible. When the hard meal, dietary, allergy, and spice rules can still be honoured, the response may separately show one `closest_numeric_fallback` with exact budget or nutrition deviations and `requires_confirmation=true`. It is never inserted into the feasible shortlist or sent to the AI ranker. Remaining-day replanning applies the same honest labelling.

## Tests

Run the deterministic tests from the repository folder:

```bash
python3 -m unittest discover -s tests -v
```

The current suite contains 43 tests. New regressions cover meal boundaries, negation, `anything` resets, meal-specific spice, portability, heaviness, avoided formats, cuisine matching, unique item sets, calorie/fat/sodium ceilings, labelled numeric fallbacks, verified user-facing explanations, strict boolean confirmation, unknown schema types, injection-heuristic bypasses and false positives, and food metadata in the OpenRouter ranking prompt.

The OpenRouter evaluation using `openai/gpt-4o-mini` produced 18 correct outcomes from 20 manually written cases (90%). E05 and E07 returned visible infeasible outcomes rather than unsuitable food. The run made 25 API calls, used 26,521 tokens, cost US$0.00525165, had no technical errors, and produced no final-plan request violations. Expected labels were not changed after seeing the output.

## Guardrails

`guardrails.py` performs deterministic checks before model calls. The injection regex is a useful early warning, not the security boundary; closed schemas, the tool allowlist, trusted server-side data, and shortlist-ID validation provide the stronger controls. The validator rejects missing fields, unknown fields, and unsupported schema types. Confirmation must be a real JSON boolean, so the string `"false"` is rejected. Optional web compatibility code namespaces idempotency keys by a user-generated session ID and calls the above-target confirmation guard before logging extra food. Invalid model output produces a visible technical error after one retry; it does not silently fall back to the optimizer.

Explicit requirements are re-derived from the original request before optimization. Python hard-filters meal-specific dietary type, spice, heaviness, portability and avoided formats, and verifies requested cuisine or format presence. If no valid option exists, Python returns infeasible before Layer 2. Layer 2 receives only verified plans and must choose one listed ID. The evaluator records exact final-plan violations rather than relying only on the pipeline outcome.

## Repository map

| Path | Purpose |
|---|---|
| `MacroFit_AI_Orchestrated_COMPLETE.ipynb` | Authoritative live demo and orchestrated evaluation |
| `core.py` | Data join, target calculation, optimizer, history and intake analysis |
| `guardrails.py` | Prompt precheck, schemas, allowlist and output validation |
| `config/` | Tool contracts and assistant policy |
| `data/menu.json` | Synthetic menu, price and preference metadata |
| `data/nutrition.json` | Synthetic nutrition and mapping provenance |
| `data/meal_rules.json` | Meal slots, times and breakfast suitability |
| `tests/` | Deterministic regression tests |
| `DATA_EXPLAINER.md` | Data methodology and limitations |
| `EVALS_EXPLAINER.md` | Evaluation design, metrics and interpretation |

## Known limitations and future work

The data is synthetic, the nutrition matching has not been independently reviewed, and session history disappears when Colab restarts. The 20-case evaluation is directional, and the five completed blind comparisons do not prove model benefit. A stronger next version would use verified outlet and nutrition records, persistent user-controlled history, explicit allergy handling, a larger set of human judgements, and a clearer egg preference field. The model should remain limited to interpretation and subjective ranking; Python should continue to own calculations and hard constraints.

None of the selected plans in the live evaluation broke the hard rules checked by Python. However, the model sometimes gave an irrelevant or unsupported explanation even when it chose a valid plan. For example, the data cannot prove that a dish is warm or comforting because it has no field for those qualities. MacroFit therefore builds the explanation shown to the user from verified menu fields. The original model explanation is kept only in the evaluation results so that this weakness remains visible.
