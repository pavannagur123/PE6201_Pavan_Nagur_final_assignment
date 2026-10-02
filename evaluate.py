from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from core import (
    ROOT,
    demo_preference_pick,
    load_eval_cases,
    load_menu,
    optimize_daily_meal_plan,
    parse_meal_preferences,
    parse_meal_spice_preferences,
)


def compact_shortlist_for_model(shortlist: list[dict]) -> list[dict]:
    """Expose real meal names and attributes; reject legacy totals-only plans."""
    compact = []
    for plan in shortlist:
        schedule = plan.get("schedule")
        if not schedule:
            raise ValueError("Evaluation requires three-meal plans with a schedule.")
        compact.append({
            "plan_id": plan["plan_id"],
            "schedule": [
                {
                    "meal": entry["meal"],
                    "item": entry["item"]["item"],
                    "dietary_type": entry["item"]["dietary_type"],
                    "spice_level": entry["item"]["spice_level"],
                    "heaviness": entry["item"]["heaviness"],
                    "meal_format": entry["item"]["meal_format"],
                    "portable": entry["item"]["portable"],
                    "quantity": entry["quantity"],
                }
                for entry in schedule
            ],
            "totals": plan["totals"],
        })
    return compact


def run_evaluation(mode: str = "optimizer") -> dict:
    """Run the fixed 20-case backend evaluation and return JSON-safe results."""
    if mode not in {"optimizer", "demo", "openrouter"}:
        raise ValueError("mode must be optimizer, demo, or openrouter")

    menu = load_menu()
    results = []
    for case in load_eval_cases():
        calories_kcal = float(case.get("calories_kcal", case["carbs_g"] * 8))
        fat_g = float(case.get("fat_g", calories_kcal * 0.30 / 9))
        meal_preferences = parse_meal_preferences(case["preferences"])
        meal_spice_preferences = parse_meal_spice_preferences(case["preferences"])
        global_spice = "spicy" if "spicy" in case["preferences"].lower() else "any"
        solution = optimize_daily_meal_plan(
            menu,
            case["budget"],
            calories_kcal,
            case["protein_g"],
            case["carbs_g"],
            fat_g=fat_g,
            meal_preferences=meal_preferences,
            meal_spice_preferences=meal_spice_preferences,
            required_spice_level=global_spice,
            shortlist_size=3,
        )
        pick = solution["optimizer_pick"]
        row = {
            "case_id": case["id"],
            "inputs": {
                "budget": case["budget"],
                "protein_g": case["protein_g"],
                "carbs_g": case["carbs_g"],
                "calories_kcal": calories_kcal,
                "fat_g": fat_g,
                "preferences": case["preferences"],
            },
            "available": solution["available"],
            "reason": solution.get("reason"),
            "optimizer_plan_id": pick["plan_id"] if pick else None,
            "optimizer_totals": pick["totals"] if pick else None,
            "constraint_checks": {},
            "feasible_plan_count": solution["feasible_count"],
            "plans_evaluated": solution["evaluated_count"],
        }
        if pick:
            totals = pick["totals"]
            row["constraint_checks"] = {
                "within_budget": totals["price_sgd"] <= case["budget"],
                "calories_between_90_and_110_percent": calories_kcal * 0.9 <= totals["kcal"] <= calories_kcal * 1.1,
                "protein_at_least_90_percent": totals["protein_g"] >= case["protein_g"] * 0.9,
                "carbs_at_least_90_percent": totals["carbs_g"] >= case["carbs_g"] * 0.9,
                "fat_at_most_120_percent": totals["fat_g"] <= fat_g * 1.2,
                "sodium_at_most_3000_mg": totals["sodium_mg"] <= 3000,
            }
        row["all_constraints_pass"] = bool(pick) and all(row["constraint_checks"].values())

        if mode == "demo" and pick and len(solution["shortlist"]) > 1:
            ranking = demo_preference_pick(solution["shortlist"], case["preferences"])
            row["preference_pick"] = ranking
        elif mode == "openrouter" and pick and len(solution["shortlist"]) > 1:
            from app import openrouter_rank

            compact_shortlist_for_model(solution["shortlist"])
            row["preference_pick"] = openrouter_rank(solution["shortlist"], case["preferences"])
        results.append(row)

    passing = sum(x["all_constraints_pass"] for x in results)
    preference_rows = [x for x in results if "preference_pick" in x]
    changed = sum(
        x["preference_pick"]["plan_id"] != x["optimizer_plan_id"]
        for x in preference_rows
    )
    return {
        "summary": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "dataset_items": len(menu),
            "dataset_outlets": len({x.shop for x in menu}),
            "origin": "NTU North Spine",
            "cases": len(results),
            "constraint_pass_count": passing,
            "constraint_pass_rate": round(passing / len(results), 4),
            "preference_layer_cases": len(preference_rows),
            "preference_pick_changed_count": changed,
            "note": "Changed picks are not wins. Human blind judgments in the notebook are required to measure preference win rate.",
        },
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MacroFit's fixed backend evaluation.")
    parser.add_argument("--mode", choices=("optimizer", "demo", "openrouter"), default="optimizer")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "evaluation" / "evaluation_results.json")
    args = parser.parse_args()
    report = run_evaluation(args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Full results: {args.output}")


if __name__ == "__main__":
    main()
