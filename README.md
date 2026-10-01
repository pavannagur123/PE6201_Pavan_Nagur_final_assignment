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

## Notebook flow

Run the notebook sequentially. It loads and validates the data, calculates the profile targets, collects the budget and request, performs the precheck, calls AI Layer 1, creates the Python shortlist, calls AI Layer 2 when subjective judgement is needed, displays the plan, and saves recommended item IDs in session history. The later cells accept an actual-intake message and recalculate the remainder of the day.

Live actual-intake clarification is interactive. Evaluation is non-interactive: a case requiring clarification is recorded with that outcome, and the evaluator moves to the next case.

Recommendation history persists only during the active Colab runtime. Set `RESET_HISTORY = True` for one run to clear it, then return it to `False`.

## Data

The prototype joins 60 synthetic menu rows from 10 synthetic outlets to 60 nutrition rows by `item_id`. The menu provides price and preference labels. Nutrition provides calories, protein, carbohydrates, fat, fibre, sodium, mapping method, source family, mapper, date, and confidence. Vegetarian means no meat, fish, or egg; egg dishes use a separate `egg` category.

The data validates the pipeline. It does not claim current prices or medically verified nutrition. See [DATA_EXPLAINER.md](DATA_EXPLAINER.md) for row definitions, checks, provenance, limitations, and replacement requirements for a real pilot.

## Evaluation and current results

The notebook contains 20 manually written cases: 12 normal requests and 8 outliers covering contradictions, ambiguous references, an infeasible budget, prompt attacks, and an extreme intake conflict. The latest recorded run produced:

| Metric | Result |
|---|---:|
| Pipeline passes | 19 of 20 |
| Pipeline pass rate | 95% |
| Layer 1 calls | 13 |
| Layer 2 calls | 12 |
| API calls | 25 |
| Evaluation cost | US$0.00546315 |
| Cost per valid judged recommendation | US$0.00109263 |
| AI and optimizer different selections | 7 of 12 Layer 2 cases |
| Blind cases judged | 5 |
| Blind outcomes | 5 same-preference ties |

The one failure was a false contradiction on “non-veg lunch” because the precheck also matched the substring “veg.” This is a known parser defect, not evidence that the optimizer or model failed. The guardrail expression must exclude `veg` when it is preceded by `non-`, followed by a rerun of all cases.

Five blind judgements are insufficient to establish that Layer 2 improves preference. Cases where AI and the optimizer select the same plan should be recorded as `same_selection`, not treated as a blind tie. Human comparison is meaningful only when the two plans differ. See [EVALS_EXPLAINER.md](EVALS_EXPLAINER.md).

## Tests

Run the deterministic tests from the repository folder:

```bash
python3 -m unittest discover -s tests -v
```

The current suite contains 33 tests covering target calculation, feasibility, meal slots, egg-free vegetarian filtering, recommendation history, shortlist diversity, actual-intake calculation, schema validation, prompt injection, model shortlist confinement, and malformed-response retry.

## Guardrails

`guardrails.py` performs deterministic checks before model calls. It blocks prompt injection, asks for clarification on same-meal dietary contradictions and missing meal references, and stops extreme restriction requests. Closed schemas reject missing or unknown fields. Python checks every selected plan ID against the approved shortlist. Confirmed consumption is kept separate from recommendations, so the notebook does not assume that displayed food was eaten.

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
