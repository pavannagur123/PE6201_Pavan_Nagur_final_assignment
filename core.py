from __future__ import annotations

import itertools
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
MENU_PATH = ROOT / "data" / "menu.json"
NUTRITION_PATH = ROOT / "data" / "nutrition.json"
EVAL_PATH = ROOT / "data" / "eval_cases.json"
MEAL_RULES_PATH = ROOT / "data" / "meal_rules.json"


@dataclass(frozen=True)
class MenuItem:
    item_id: str
    shop: str
    item: str
    cuisine: str
    price_sgd: float
    location_name: str
    origin_name: str
    distance_km: float
    walk_min: int
    dietary_type: str
    spice_level: str
    heaviness: str
    meal_format: str
    portable: bool
    preference_tags: list[str]
    preference_metadata_status: str
    kcal: int
    protein_g: float
    carbs_g: float
    fat_g: float
    fibre_g: float
    sodium_mg: int
    source_family: str
    source_match: str
    mapping_method: str
    mapped_by: str
    mapped_on: str
    confidence: str


def load_menu(menu_path: Path = MENU_PATH, nutrition_path: Path = NUTRITION_PATH) -> list[MenuItem]:
    """Join the price/location dataset to the nutrition dataset by item_id."""
    nutrition_rows = json.loads(nutrition_path.read_text(encoding="utf-8"))
    nutrition = {row["item_id"]: row for row in nutrition_rows}

    rows = []
    for menu_row in json.loads(menu_path.read_text(encoding="utf-8")):
        item_id = menu_row["item_id"]
        if item_id not in nutrition:
            raise ValueError(f"Missing nutrition row for {item_id}.")
        rows.append(MenuItem(**{**menu_row, **nutrition[item_id]}))

    menu_ids = {x.item_id for x in rows}
    unused_nutrition = set(nutrition) - menu_ids
    if unused_nutrition:
        raise ValueError(f"Nutrition rows without menu records: {sorted(unused_nutrition)}")
    return rows


def load_eval_cases(path: Path = EVAL_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_meal_rules(path: Path = MEAL_RULES_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,
    "lightly_active": 1.375,
    "moderately_active": 1.55,
    "very_active": 1.725,
    "extra_active": 1.9,
}

GOAL_ADJUSTMENTS = {
    "lose": -300,
    "maintain": 0,
    "gain": 300,
}


def calculate_daily_targets(
    age: int,
    weight_kg: float,
    height_cm: float,
    calculation_sex: str,
    activity_level: str,
    goal: str,
) -> dict:
    """Estimate adult wellness targets with deterministic, auditable math."""
    if not 18 <= age <= 100:
        raise ValueError("Age must be between 18 and 100 for this adult-only prototype.")
    if not 35 <= weight_kg <= 300 or not 120 <= height_cm <= 230:
        raise ValueError("Weight or height is outside the supported prototype range.")
    if calculation_sex not in {"female", "male"}:
        raise ValueError("Calculation sex must be female or male.")
    if activity_level not in ACTIVITY_MULTIPLIERS or goal not in GOAL_ADJUSTMENTS:
        raise ValueError("Activity level or goal is not supported.")

    sex_constant = 5 if calculation_sex == "male" else -161
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + sex_constant
    maintenance = bmr * ACTIVITY_MULTIPLIERS[activity_level]
    calories = max(1200, maintenance + GOAL_ADJUSTMENTS[goal])

    # Balanced general-wellness default within adult AMDR ranges:
    # 20% protein, 50% carbohydrate, 30% fat.
    protein_g = calories * 0.20 / 4
    carbs_g = calories * 0.50 / 4
    fat_g = calories * 0.30 / 9
    return {
        "estimated_bmr_kcal": round(bmr),
        "estimated_maintenance_kcal": round(maintenance),
        "estimated_daily_calories_kcal": round(calories),
        "protein_target_g": round(protein_g),
        "carb_target_g": round(carbs_g),
        "fat_target_g": round(fat_g),
        "calculation_method": "Mifflin-St Jeor; activity multiplier; goal adjustment; 20/50/30 macro split",
        "medical_advice": False,
    }


def _totals(items: Iterable[MenuItem]) -> dict:
    items = tuple(items)
    return {
        "price_sgd": round(sum(x.price_sgd for x in items), 2),
        "kcal": sum(x.kcal for x in items),
        "protein_g": round(sum(x.protein_g for x in items), 1),
        "carbs_g": round(sum(x.carbs_g for x in items), 1),
        "fat_g": round(sum(x.fat_g for x in items), 1),
        "fibre_g": round(sum(x.fibre_g for x in items), 1),
        "sodium_mg": sum(x.sodium_mg for x in items),
        "max_distance_km": round(max((x.distance_km for x in items), default=0), 1),
        "max_walk_min": max((x.walk_min for x in items), default=0),
    }


def _score(totals: dict, budget: float, protein: float, carbs: float, shops: int) -> tuple:
    protein_floor = protein * 0.9
    carbs_floor = carbs * 0.9
    budget_gap = max(0.0, totals["price_sgd"] - budget) / max(budget, 1)
    protein_gap = max(0.0, protein_floor - totals["protein_g"]) / max(protein_floor, 1)
    carbs_gap = max(0.0, carbs_floor - totals["carbs_g"]) / max(carbs_floor, 1)
    feasible = budget_gap == protein_gap == carbs_gap == 0

    # Once feasible, prefer closeness and leave a little budget headroom.
    protein_distance = abs(totals["protein_g"] - protein) / max(protein, 1)
    carbs_distance = abs(totals["carbs_g"] - carbs) / max(carbs, 1)
    budget_use = totals["price_sgd"] / max(budget, 1)
    multi_shop_penalty = max(0, shops - 1) * 0.015
    if feasible:
        distance = protein_distance * 0.49 + carbs_distance * 0.38 + budget_use * 0.115 + multi_shop_penalty
    else:
        distance = 10 + budget_gap * 5 + protein_gap * 3 + carbs_gap * 3 + multi_shop_penalty
    return feasible, round(distance, 5), {
        "budget_gap": round(budget_gap, 4),
        "protein_gap": round(protein_gap, 4),
        "carbs_gap": round(carbs_gap, 4),
    }


def optimize(menu: list[MenuItem], budget: float, protein_g: float, carbs_g: float, shortlist_size: int = 5) -> dict:
    if not 3 <= budget <= 100:
        raise ValueError("Budget must be between S$3 and S$100.")
    if not 10 <= protein_g <= 250 or not 20 <= carbs_g <= 500:
        raise ValueError("Macro targets are outside the supported prototype range.")

    candidates = []
    seen = set()
    for count in range(1, 4):
        for combo in itertools.combinations(menu, count):
            ids = tuple(sorted(item.item_id for item in combo))
            if ids in seen:
                continue
            seen.add(ids)
            totals = _totals(combo)
            feasible, score, gaps = _score(totals, budget, protein_g, carbs_g, len({x.shop for x in combo}))
            candidates.append({
                "plan_id": "+".join(ids),
                "items": [asdict(x) for x in combo],
                "totals": totals,
                "feasible": feasible,
                "score": score,
                "gaps": gaps,
            })

    candidates.sort(key=lambda x: (not x["feasible"], x["score"], x["totals"]["price_sgd"], x["plan_id"]))
    shortlist = candidates[:shortlist_size]
    return {
        "inputs": {"budget": budget, "protein_g": protein_g, "carbs_g": carbs_g, "threshold": 0.9},
        "feasible_count": sum(1 for x in candidates if x["feasible"]),
        "evaluated_count": len(candidates),
        "shortlist": shortlist,
        "optimizer_pick": shortlist[0],
    }


def parse_meal_preferences(text: str, explicit: dict | None = None) -> dict:
    """Parse meal-local dietary wording without leaking across meal boundaries.

    This is intentionally narrow. Negated food words do not become positive
    requirements, and an explicit ``anything`` resets only the named meal.
    Flexible or unclear language belongs to the structured AI interpreter.
    """
    meals = ("breakfast", "lunch", "dinner")
    result = {meal: (explicit or {}).get(meal, "any") for meal in meals}
    lowered = re.sub(r"\s+", " ", str(text or "").lower()).strip()

    global_vegetarian = re.search(
        r"\b(?:veg|vegetarian|meat[- ]?free|meatless)\b.{0,20}\b(?:all meals|all day|everything)\b",
        lowered,
    )
    if global_vegetarian:
        for meal in meals:
            if result[meal] == "any":
                result[meal] = "vegetarian"

    clauses = [part.strip() for part in re.split(r"[,;.]|\b(?:and|but)\b", lowered) if part.strip()]
    for clause in clauses:
        meal_matches = list(re.finditer(r"\b(?:breakfast|lunch|dinner)\b", clause))
        for index, meal_match in enumerate(meal_matches):
            meal = meal_match.group(0)
            if (explicit or {}).get(meal, "any") != "any":
                continue
            left = 0 if index == 0 else (meal_matches[index - 1].end() + meal_match.start()) // 2
            right = len(clause) if index == len(meal_matches) - 1 else (meal_match.end() + meal_matches[index + 1].start()) // 2
            segment = clause[left:right]

            if re.search(r"\b(?:anything|any|whatever|no preference)\b", segment):
                result[meal] = "any"
                continue

            excludes_chicken = bool(re.search(r"\b(?:no|not|avoid|without)\s+(?:any\s+)?chicken\b", segment))
            cleaned = re.sub(
                r"\b(?:no|not|avoid|without)\s+(?:any\s+)?(?:chicken|veg|vegetarian)\b",
                "",
                segment,
            )
            wants_vegetarian = bool(re.search(
                r"(?<!non-)(?<!non )\b(?:veg|vegetarian|meat[- ]?free|meatless)\b|\bno\s+meat\b",
                cleaned,
            ))
            wants_chicken = bool(re.search(r"\b(?:chicken|non[- ]?veg(?:etarian)?)\b", cleaned))
            if wants_vegetarian != wants_chicken:
                result[meal] = "vegetarian" if wants_vegetarian else "chicken"
            elif excludes_chicken:
                result[meal] = "not_chicken"
    return result


def parse_meal_spice_preferences(text: str) -> dict:
    """Return explicit meal-specific spice requirements; never infer a meal."""
    result = {meal: "any" for meal in ("breakfast", "lunch", "dinner")}
    lowered = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    clauses = [part.strip() for part in re.split(r"[,;.]|\b(?:and|but)\b", lowered) if part.strip()]
    for clause in clauses:
        meal_matches = list(re.finditer(r"\b(?:breakfast|lunch|dinner)\b", clause))
        for index, meal_match in enumerate(meal_matches):
            meal = meal_match.group(0)
            left = 0 if index == 0 else (meal_matches[index - 1].end() + meal_match.start()) // 2
            right = len(clause) if index == len(meal_matches) - 1 else (meal_match.end() + meal_matches[index + 1].start()) // 2
            segment = clause[left:right]
            for level in ("spicy", "medium", "mild"):
                if re.search(rf"\b{level}\b", segment) and not re.search(rf"\b(?:not|avoid|no)\s+{level}\b", segment):
                    result[meal] = level
                    break
    return result


def parse_explicit_food_requirements(text: str) -> dict:
    """Extract requirements that must be verified against dataset metadata.

    The language model may add subjective interpretation, but these explicit
    meal-local and negative requirements are deterministic guardrails. This
    prevents a later model call from describing an unsuitable plan as a match.
    """
    meals = ("breakfast", "lunch", "dinner")
    lowered = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    dietary = parse_meal_preferences(lowered)
    spice = parse_meal_spice_preferences(lowered)
    requirements = {
        meal: {
            "dietary_type": dietary[meal],
            "spice_level": spice[meal],
            "heaviness": "any",
            "portability": "any",
            "avoided_formats": [],
        }
        for meal in meals
    }
    clauses = [part.strip() for part in re.split(r"[,;.]|\b(?:and|but)\b", lowered) if part.strip()]
    for clause in clauses:
        mentioned_meals = re.findall(r"\b(?:breakfast|lunch|dinner)\b", clause)
        for meal in mentioned_meals:
            for value in ("light", "balanced", "hearty"):
                if re.search(rf"\b{value}\b", clause):
                    requirements[meal]["heaviness"] = value
            if re.search(r"\bportable\b", clause):
                requirements[meal]["portability"] = "portable"

    global_avoided_formats = []
    if re.search(r"\b(?:no|not|avoid|without|nothing)\b.{0,18}\b(?:rice|rice bowl|rice bowls)\b", lowered):
        global_avoided_formats.append("rice_or_bowl")
    if re.search(r"\b(?:no|not|avoid|without|nothing)\b.{0,18}\b(?:soup|soupy)\b", lowered):
        global_avoided_formats.append("soupy")

    global_portability = "portable" if re.search(r"\bportable\s+(?:food|meals?)\b", lowered) else "any"
    global_spice = "any"
    for value in ("spicy", "medium", "mild"):
        if re.search(rf"\b{value}\s+(?:food|meals?)\b", lowered):
            global_spice = value
            break

    required_any_formats = []
    if re.search(r"\b(?:prefer|want|feel like)\b.{0,20}\bnoodles?\b", lowered):
        required_any_formats.append("noodles")

    cuisine_aliases = {
        "indian": "Indian", "chinese": "Chinese", "korean": "Korean",
        "malay": "Malay", "western": "Western", "mediterranean": "Mediterranean",
        "japanese": "Japanese", "thai": "Thai",
    }
    required_any_cuisines = [label for token, label in cuisine_aliases.items() if re.search(rf"\b{token}\b", lowered)]
    return {
        "meals": requirements,
        "global_avoided_formats": sorted(set(global_avoided_formats)),
        "global_portability": global_portability,
        "global_spice_level": global_spice,
        "required_any_formats": sorted(set(required_any_formats)),
        "required_any_cuisines": sorted(set(required_any_cuisines)),
        "no_repeated_food": bool(re.search(r"\b(?:no repeated|no repeats?|variety)\b", lowered)),
    }


def plan_requirement_violations(plan: dict, requirements: dict) -> list[str]:
    """Return exact metadata contradictions between a plan and the request."""
    violations = []
    schedule = plan.get("schedule", [])
    by_meal = {entry.get("meal"): entry.get("item", entry) for entry in schedule}
    for meal, expected in requirements.get("meals", {}).items():
        item = by_meal.get(meal)
        if not item:
            violations.append(f"missing_{meal}")
            continue
        dietary = expected.get("dietary_type", "any")
        actual_dietary = item.get("dietary_type")
        if dietary == "not_chicken" and actual_dietary == "chicken":
            violations.append(f"{meal}_must_not_be_chicken")
        elif dietary not in {"any", "not_chicken"} and actual_dietary != dietary:
            violations.append(f"{meal}_dietary_type_expected_{dietary}_got_{actual_dietary}")
        for field in ("spice_level", "heaviness"):
            wanted = expected.get(field, "any")
            if wanted != "any" and item.get(field) != wanted:
                violations.append(f"{meal}_{field}_expected_{wanted}_got_{item.get(field)}")
        if expected.get("portability") == "portable" and not item.get("portable"):
            violations.append(f"{meal}_must_be_portable")
        if item.get("meal_format") in set(expected.get("avoided_formats", [])):
            violations.append(f"{meal}_format_{item.get('meal_format')}_is_avoided")

    items = list(by_meal.values())
    avoided = set(requirements.get("global_avoided_formats", []))
    if avoided:
        for meal, item in by_meal.items():
            if item.get("meal_format") in avoided:
                violations.append(f"{meal}_format_{item.get('meal_format')}_is_avoided")
    if requirements.get("global_portability") == "portable":
        for meal, item in by_meal.items():
            if not item.get("portable"):
                violations.append(f"{meal}_must_be_portable")
    global_spice = requirements.get("global_spice_level", "any")
    if global_spice != "any":
        for meal, item in by_meal.items():
            if item.get("spice_level") != global_spice:
                violations.append(f"{meal}_spice_level_expected_{global_spice}_got_{item.get('spice_level')}")
    formats = set(requirements.get("required_any_formats", []))
    if formats and not any(item.get("meal_format") in formats for item in items):
        violations.append("plan_missing_required_format_" + "_or_".join(sorted(formats)))
    cuisines = {value.lower() for value in requirements.get("required_any_cuisines", [])}
    if cuisines and not any(str(item.get("cuisine", "")).lower() in cuisines for item in items):
        violations.append("plan_missing_required_cuisine_" + "_or_".join(sorted(cuisines)))
    names = [re.sub(r"\s+", " ", str(item.get("item", "")).lower()).strip() for item in items]
    if requirements.get("no_repeated_food") and len(names) != len(set(names)):
        violations.append("plan_repeats_food")
    return sorted(set(violations))


def verified_plan_explanation(plan: dict, requirements: dict) -> str:
    """Build a factual user-facing explanation from trusted plan metadata."""
    violations = plan_requirement_violations(plan, requirements)
    if violations:
        raise ValueError("Cannot explain a plan that violates the request: " + ", ".join(violations))
    by_meal = {entry["meal"]: entry.get("item", entry) for entry in plan.get("schedule", [])}
    facts = []
    for meal in ("breakfast", "lunch", "dinner"):
        item = by_meal.get(meal)
        expected = requirements.get("meals", {}).get(meal, {})
        if not item:
            continue
        details = []
        dietary = expected.get("dietary_type", "any")
        if dietary == "not_chicken":
            details.append("no chicken")
        elif dietary != "any":
            details.append(dietary)
        spice = expected.get("spice_level", "any")
        if spice != "any":
            details.append(spice)
        heaviness = expected.get("heaviness", "any")
        if heaviness != "any":
            details.append(heaviness)
        if expected.get("portability") == "portable":
            details.append("portable")
        if details:
            facts.append(f"{meal.title()} is {', '.join(details)}: {item['item']}")
    if requirements.get("global_avoided_formats"):
        labels = ", ".join(requirements["global_avoided_formats"])
        facts.append(f"The plan avoids the requested formats: {labels}")
    if requirements.get("global_portability") == "portable":
        facts.append("All three meals are marked portable in the dataset")
    if requirements.get("required_any_formats"):
        facts.append("The plan includes the requested format: " + ", ".join(requirements["required_any_formats"]))
    if requirements.get("required_any_cuisines"):
        facts.append("The plan includes the requested cuisine: " + ", ".join(requirements["required_any_cuisines"]))
    if requirements.get("no_repeated_food"):
        facts.append("All three dishes are different")
    if not facts:
        facts.append("Every meal passed the budget, nutrition, meal-slot and explicit request checks")
    return ". ".join(facts) + "."


def _items_for_meal(
    menu: list[MenuItem],
    rule: dict,
    preference: str = "any",
    excluded_item_ids: set[str] | frozenset[str] | None = None,
    spice_preference: str = "any",
    explicit_requirement: dict | None = None,
    global_avoided_formats: set[str] | frozenset[str] | None = None,
    global_portability: str = "any",
    global_spice_level: str = "any",
) -> list[MenuItem]:
    allowed = set(rule.get("allowed_item_ids", []))
    excluded = set(rule.get("excluded_item_ids", []))
    excluded.update(excluded_item_ids or ())
    maximum_kcal = rule.get("maximum_kcal")
    explicit_requirement = explicit_requirement or {}
    avoided_formats = set(explicit_requirement.get("avoided_formats", [])) | set(global_avoided_formats or ())
    required_heaviness = explicit_requirement.get("heaviness", "any")
    required_portability = explicit_requirement.get("portability", "any")
    effective_spice = explicit_requirement.get("spice_level", spice_preference)
    if effective_spice == "any":
        effective_spice = global_spice_level
    return [
        item for item in menu
        if (not allowed or item.item_id in allowed)
        and item.item_id not in excluded
        and (maximum_kcal is None or item.kcal <= maximum_kcal)
        and (preference != "vegetarian" or item.dietary_type == "vegetarian")
        and (preference != "chicken" or item.dietary_type == "chicken")
        and (preference != "not_chicken" or item.dietary_type != "chicken")
        and (effective_spice == "any" or item.spice_level == effective_spice)
        and (required_heaviness == "any" or item.heaviness == required_heaviness)
        and (required_portability != "portable" or item.portable)
        and (global_portability != "portable" or item.portable)
        and item.meal_format not in avoided_formats
    ]


def _weighted_totals(choices: tuple[tuple[MenuItem, float], ...]) -> dict:
    return {
        "price_sgd": round(sum(item.price_sgd * quantity for item, quantity in choices), 2),
        "kcal": round(sum(item.kcal * quantity for item, quantity in choices)),
        "protein_g": round(sum(item.protein_g * quantity for item, quantity in choices), 1),
        "carbs_g": round(sum(item.carbs_g * quantity for item, quantity in choices), 1),
        "fat_g": round(sum(item.fat_g * quantity for item, quantity in choices), 1),
        "fibre_g": round(sum(item.fibre_g * quantity for item, quantity in choices), 1),
        "sodium_mg": round(sum(item.sodium_mg * quantity for item, quantity in choices)),
        "max_distance_km": round(max(item.distance_km for item, _ in choices), 1),
        "max_walk_min": max(item.walk_min for item, _ in choices),
    }


def minimum_three_meal_budget(menu: list[MenuItem], rules: dict | None = None) -> float:
    """Return the cheapest valid, distinct breakfast/lunch/dinner combination."""
    rules = rules or load_meal_rules()
    schedule = rules["schedule"]
    pools = [_items_for_meal(menu, schedule[name]) for name in ("breakfast", "lunch", "dinner")]
    prices = [
        sum(item.price_sgd for item in combo)
        for combo in itertools.product(*pools)
        if len({item.item_id for item in combo}) == 3
    ]
    if not prices:
        raise ValueError("The dataset has no valid breakfast, lunch, and dinner combination.")
    return round(min(prices), 2)


def optimize_daily_meal_plan(
    menu: list[MenuItem],
    budget: float,
    calories_kcal: float,
    protein_g: float,
    carbs_g: float,
    shortlist_size: int = 3,
    rules: dict | None = None,
    meal_preferences: dict | None = None,
    excluded_item_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    recent_item_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    fat_g: float | None = None,
    calorie_ceiling_ratio: float = 1.10,
    fat_ceiling_ratio: float = 1.20,
    sodium_ceiling_mg: float = 3000,
    meal_spice_preferences: dict | None = None,
    required_spice_level: str = "any",
    explicit_requirements: dict | None = None,
) -> dict:
    """Build three meals with bounded calories, fat, sodium and diverse item sets."""
    rules = rules or load_meal_rules()
    minimum_budget = minimum_three_meal_budget(menu, rules)
    schedule_rules = rules["schedule"]
    meal_names = ("breakfast", "lunch", "dinner")
    meal_preferences = meal_preferences or {name: "any" for name in meal_names}
    meal_spice_preferences = meal_spice_preferences or {name: "any" for name in meal_names}
    explicit_requirements = explicit_requirements or {
        "meals": {name: {} for name in meal_names},
        "global_avoided_formats": [],
        "global_portability": "any",
        "global_spice_level": "any",
        "required_any_formats": [],
        "required_any_cuisines": [],
        "no_repeated_food": False,
    }
    if required_spice_level not in {"any", "mild", "medium", "spicy"}:
        raise ValueError("Required spice level must be any, mild, medium, or spicy.")
    if any(meal_spice_preferences.get(name, "any") not in {"any", "mild", "medium", "spicy"} for name in meal_names):
        raise ValueError("Meal spice preferences must be any, mild, medium, or spicy.")
    excluded_item_ids = frozenset(excluded_item_ids or ())
    recent_item_ids = frozenset(recent_item_ids or ())
    known_item_ids = {item.item_id for item in menu}
    unknown_history_ids = (excluded_item_ids | recent_item_ids) - known_item_ids
    if unknown_history_ids:
        raise ValueError(f"Unknown history item IDs: {sorted(unknown_history_ids)}")
    if any(meal_preferences.get(name, "any") not in {"any", "vegetarian", "chicken", "not_chicken"} for name in meal_names):
        raise ValueError("Meal preferences must be any, vegetarian, chicken, or not_chicken.")
    item_pools = [
        _items_for_meal(
            menu,
            schedule_rules[name],
            meal_preferences.get(name, "any"),
            excluded_item_ids,
            meal_spice_preferences.get(name, "any"),
            explicit_requirements.get("meals", {}).get(name, {}),
            set(explicit_requirements.get("global_avoided_formats", [])),
            explicit_requirements.get("global_portability", "any"),
            explicit_requirements.get("global_spice_level", "any"),
        )
        for name in meal_names
    ]
    if any(not pool for pool in item_pools):
        return {
            "available": False,
            "minimum_budget_sgd": minimum_budget,
            "minimum_target_budget_sgd": None,
            "message": "No menu items satisfy the selected meal-specific preferences.",
            "reason": "meal_preference_unavailable",
            "calorie_shortfall_kcal": None,
            "suggested_minimum_budget_sgd": None,
            "evaluated_count": 0,
            "feasible_count": 0,
            "shortlist": [],
            "optimizer_pick": None,
            "closest_fallback": None,
        }

    quantity_pools = [
        [(item, 1.0) for item in item_pools[0]],
        [(item, quantity) for item in item_pools[1] for quantity in (1.0, 1.5, 2.0)],
        [(item, quantity) for item in item_pools[2] for quantity in (1.0, 1.5, 2.0)],
    ]
    best = []
    closest_fallback_row = None
    evaluated_count = 0
    feasible_count = 0
    minimum_target_budget = None
    best_calories_within_budget = 0
    calorie_floor = calories_kcal * 0.9
    calorie_ceiling = calories_kcal * calorie_ceiling_ratio
    protein_floor = protein_g * 0.9
    carbs_floor = carbs_g * 0.9
    fat_g = float(fat_g if fat_g is not None else calories_kcal * 0.30 / 9)
    fat_ceiling = fat_g * fat_ceiling_ratio
    required_any_formats = set(explicit_requirements.get("required_any_formats", []))
    required_any_cuisines = {
        str(value).lower() for value in explicit_requirements.get("required_any_cuisines", [])
    }
    require_unique_names = bool(explicit_requirements.get("no_repeated_food"))
    for choices in itertools.product(*quantity_pools):
        combo = tuple(item for item, _ in choices)
        if len({item.item_id for item in combo}) != 3:
            continue
        evaluated_count += 1
        totals = _weighted_totals(choices)
        if totals["price_sgd"] <= budget:
            best_calories_within_budget = max(best_calories_within_budget, totals["kcal"])
        has_required_spice = (
            required_spice_level == "any"
            or any(item.spice_level == required_spice_level for item, _ in choices)
        )
        if not has_required_spice:
            continue
        if required_any_formats and not any(item.meal_format in required_any_formats for item in combo):
            continue
        if required_any_cuisines and not any(item.cuisine.lower() in required_any_cuisines for item in combo):
            continue
        if require_unique_names:
            normalized_names = {re.sub(r"\s+", " ", item.item.lower()).strip() for item in combo}
            if len(normalized_names) != len(combo):
                continue
        violations = {
            "budget_excess_sgd": round(max(0, totals["price_sgd"] - budget), 2),
            "calorie_shortfall_kcal": round(max(0, calorie_floor - totals["kcal"])),
            "calorie_excess_kcal": round(max(0, totals["kcal"] - calorie_ceiling)),
            "protein_shortfall_g": round(max(0, protein_floor - totals["protein_g"]), 1),
            "carb_shortfall_g": round(max(0, carbs_floor - totals["carbs_g"]), 1),
            "fat_excess_g": round(max(0, totals["fat_g"] - fat_ceiling), 1),
            "sodium_excess_mg": round(max(0, totals["sodium_mg"] - sodium_ceiling_mg)),
        }
        fallback_distance = (
            violations["budget_excess_sgd"] / max(budget, 1) * 2.0
            + violations["calorie_shortfall_kcal"] / max(calories_kcal, 1)
            + violations["calorie_excess_kcal"] / max(calories_kcal, 1)
            + violations["protein_shortfall_g"] / max(protein_g, 1)
            + violations["carb_shortfall_g"] / max(carbs_g, 1)
            + violations["fat_excess_g"] / max(fat_g, 1)
            + violations["sodium_excess_mg"] / max(sodium_ceiling_mg, 1)
        )
        fallback_row = (round(fallback_distance, 6), totals["price_sgd"], choices, totals, violations)
        if closest_fallback_row is None or fallback_row[:2] < closest_fallback_row[:2]:
            closest_fallback_row = fallback_row
        meets_targets = not any(
            value for key, value in violations.items() if key != "budget_excess_sgd"
        )
        if meets_targets:
            minimum_target_budget = totals["price_sgd"] if minimum_target_budget is None else min(minimum_target_budget, totals["price_sgd"])
        if not meets_targets or totals["price_sgd"] > budget:
            continue
        feasible_count += 1
        recent_item_penalty = sum(
            0.08 for item, _ in choices if item.item_id in recent_item_ids
        )
        score = (
            abs(totals["kcal"] - calories_kcal) / calories_kcal * 0.64
            + abs(totals["protein_g"] - protein_g) / protein_g * 0.16
            + abs(totals["carbs_g"] - carbs_g) / carbs_g * 0.14
            + totals["price_sgd"] / budget * 0.06
            + max(0, totals["sodium_mg"] - 2000) / 2000 * 0.08
            + recent_item_penalty
        )
        best.append((round(score, 5), totals["price_sgd"], choices, totals))
        candidate_pool_size = max(100, shortlist_size * 20)
        if len(best) > candidate_pool_size * 2:
            best.sort(key=lambda row: (row[0], row[1], tuple(item.item_id for item, _ in row[2])))
            del best[candidate_pool_size:]

    best.sort(key=lambda row: (row[0], row[1], tuple(item.item_id for item, _ in row[2])))
    # Meal permutations of the same dishes are not meaningfully different A/B
    # choices. Keep only the numerically best schedule for each item set.
    unique_item_sets = []
    seen_item_sets = set()
    for row in best:
        item_set = frozenset(item.item_id for item, _ in row[2])
        if item_set in seen_item_sets:
            continue
        seen_item_sets.add(item_set)
        unique_item_sets.append(row)
    best = unique_item_sets
    # Keep the numerically best plan first, then deliberately diversify the
    # shortlist so the AI sees meaningfully different food and cuisines.
    diverse_best = []
    remaining = list(best[:max(100, shortlist_size * 20)])
    if remaining:
        diverse_best.append(remaining.pop(0))
    while remaining and len(diverse_best) < shortlist_size:
        def diversity_rank(row):
            row_ids = {item.item_id for item, _ in row[2]}
            row_cuisines = {item.cuisine for item, _ in row[2]}
            maximum_item_overlap = max(
                len(row_ids & {item.item_id for item, _ in selected[2]})
                for selected in diverse_best
            )
            maximum_cuisine_overlap = max(
                len(row_cuisines & {item.cuisine for item, _ in selected[2]})
                for selected in diverse_best
            )
            adjusted_score = row[0] + maximum_item_overlap * 0.05 + maximum_cuisine_overlap * 0.015
            return (adjusted_score, row[0], row[1], tuple(item.item_id for item, _ in row[2]))
        chosen = min(remaining, key=diversity_rank)
        diverse_best.append(chosen)
        remaining.remove(chosen)

    shortlist = []
    for score, _, choices, totals in diverse_best:
        schedule = []
        for meal, (item, quantity) in zip(meal_names, choices):
            rule = schedule_rules[meal]
            schedule.append({
                "meal": meal,
                "label": rule["label"],
                "eat_at": rule["eat_at"],
                "window": rule["window"],
                "quantity": quantity,
                "item": asdict(item),
            })
        shortlist.append({
            "plan_id": "+".join(f"{item.item_id}x{quantity:g}" for item, quantity in choices),
            "items": [asdict(item) for item, _ in choices],
            "schedule": schedule,
            "totals": totals,
            "feasible": True,
            "score": score,
            "gaps": {"calorie_gap": 0, "protein_gap": 0, "carb_gap": 0, "fat_excess": 0, "sodium_excess": 0, "budget_gap": 0},
        })

    if not shortlist:
        if budget < minimum_budget:
            message = f"No valid three-meal day fits S${budget:.2f}; the minimum meal-slot-valid budget is S${minimum_budget:.2f}."
        elif minimum_target_budget is None:
            message = "No three-meal combination in the current dataset reaches 90% of the calculated calorie and macro estimates."
        else:
            message = f"No plan can reach 90% of the calculated targets within S${budget:.2f}. Increase the budget to at least S${minimum_target_budget:.2f}."
        closest_fallback = None
        if closest_fallback_row is not None:
            fallback_score, _, fallback_choices, fallback_totals, violations = closest_fallback_row
            fallback_schedule = []
            for meal, (item, quantity) in zip(meal_names, fallback_choices):
                rule = schedule_rules[meal]
                fallback_schedule.append({
                    "meal": meal,
                    "label": rule["label"],
                    "eat_at": rule["eat_at"],
                    "window": rule["window"],
                    "quantity": quantity,
                    "item": asdict(item),
                })
            closest_fallback = {
                "selection_type": "closest_numeric_fallback",
                "plan_id": "+".join(f"{item.item_id}x{quantity:g}" for item, quantity in fallback_choices),
                "schedule": fallback_schedule,
                "totals": fallback_totals,
                "violations": {key: value for key, value in violations.items() if value},
                "distance_score": fallback_score,
                "feasible": False,
                "requires_confirmation": True,
                "warning": "This is the closest dataset-backed option, not a plan that meets all numeric limits.",
            }
            message += " A labelled closest option is available for review, but it requires explicit confirmation."
        reason = (
            "budget_below_three_meal_minimum"
            if budget < minimum_budget
            else "targets_unavailable_within_budget" if minimum_target_budget is not None
            else "targets_unavailable_in_dataset"
        )
        return {
            "available": False,
            "minimum_budget_sgd": minimum_budget,
            "minimum_target_budget_sgd": minimum_target_budget,
            "message": message,
            "reason": reason,
            "calorie_shortfall_kcal": max(0, round(calorie_floor - best_calories_within_budget)),
            "suggested_minimum_budget_sgd": minimum_target_budget,
            "evaluated_count": evaluated_count,
            "feasible_count": 0,
            "shortlist": [],
            "optimizer_pick": None,
            "closest_fallback": closest_fallback,
        }
    return {
        "available": True,
        "minimum_budget_sgd": minimum_budget,
        "minimum_target_budget_sgd": round(minimum_target_budget, 2),
        "message": "",
        "inputs": {
            "budget": budget,
            "calories_kcal": calories_kcal,
            "protein_g": protein_g,
            "carbs_g": carbs_g,
            "threshold": 0.9,
            "calorie_ceiling_ratio": calorie_ceiling_ratio,
            "fat_g": fat_g,
            "fat_ceiling_ratio": fat_ceiling_ratio,
            "sodium_ceiling_mg": sodium_ceiling_mg,
            "meal_preferences": meal_preferences,
            "meal_spice_preferences": meal_spice_preferences,
            "required_spice_level": required_spice_level,
            "excluded_item_ids": sorted(excluded_item_ids),
            "recent_item_ids": sorted(recent_item_ids),
            "explicit_requirements": explicit_requirements,
        },
        "evaluated_count": evaluated_count,
        "feasible_count": feasible_count,
        "shortlist": shortlist,
        "optimizer_pick": shortlist[0],
        "closest_fallback": None,
    }


def analyze_actual_intake(menu: list[MenuItem], actual_items: list[dict], targets: dict, budget_sgd: float) -> dict:
    """Compare confirmed consumption with the deterministic daily estimate."""
    menu_by_id = {item.item_id: item for item in menu}
    totals = {"price_sgd": 0.0, "kcal": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
    for entry in actual_items:
        item = menu_by_id.get(entry["item_id"])
        if item is None:
            raise ValueError(f"Unknown menu item: {entry['item_id']}.")
        quantity = float(entry["quantity"])
        if not 0 < quantity <= 10:
            raise ValueError("Quantity must be greater than 0 and at most 10.")
        totals["price_sgd"] += item.price_sgd * quantity
        totals["kcal"] += item.kcal * quantity
        totals["protein_g"] += item.protein_g * quantity
        totals["carbs_g"] += item.carbs_g * quantity
        totals["fat_g"] += item.fat_g * quantity

    totals = {key: round(value, 1 if key != "price_sgd" else 2) for key, value in totals.items()}
    calorie_target = float(targets["calories_kcal"])
    calorie_difference = round(totals["kcal"] - calorie_target)
    if calorie_difference > 0:
        status = "above_estimate"
        message = f"Your logged food is about {calorie_difference} kcal above today's estimate."
    elif calorie_difference < 0:
        status = "below_estimate"
        message = f"Your logged food is about {abs(calorie_difference)} kcal below today's estimate."
    else:
        status = "at_estimate"
        message = "Your logged food matches today's calorie estimate."
    return {
        "status": status,
        "message": message,
        "actual_totals": totals,
        "differences": {
            "calories_kcal": calorie_difference,
            "protein_g": round(totals["protein_g"] - float(targets["protein_g"]), 1),
            "carbs_g": round(totals["carbs_g"] - float(targets["carbs_g"]), 1),
            "fat_g": round(totals["fat_g"] - float(targets["fat_g"]), 1),
            "budget_sgd": round(totals["price_sgd"] - budget_sgd, 2),
        },
        "within_budget": totals["price_sgd"] <= budget_sgd,
        "medical_advice": False,
    }


def demo_preference_pick(shortlist: list[dict], preferences: str) -> dict:
    """Transparent, non-LLM demo only; never count this as model evidence."""
    text = preferences.lower()
    scored = []
    for plan in shortlist:
        blob = " ".join(f"{x['item']} {x['cuisine']} {x['shop']}" for x in plan["items"]).lower()
        bonus = 0
        for token in ("chicken", "vegetarian", "indian", "chinese", "malay", "western", "rice", "noodle", "spicy", "fish"):
            if token in text and token in blob:
                bonus += 1
        if "variety" in text:
            bonus += len({x["cuisine"] for x in plan["items"]}) * 0.35
        scored.append((bonus, -plan["score"], plan))
    chosen = max(scored, key=lambda x: (x[0], x[1]))[2]
    return {
        "plan_id": chosen["plan_id"],
        "reason": "Demo heuristic matched stated food words and variety, then used optimizer score as a tie-breaker.",
        "mode": "demo_heuristic",
    }
