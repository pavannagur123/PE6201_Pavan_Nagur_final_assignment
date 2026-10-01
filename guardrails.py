from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core import ROOT, load_menu


DESCRIPTORS_PATH = ROOT / "config" / "tool_descriptors.json"
POLICY_PATH = ROOT / "config" / "agent_guardrails.json"
MAX_TOOL_ARGUMENT_BYTES = 12_000


@dataclass
class GuardrailViolation(ValueError):
    code: str
    safe_message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.safe_message}"


@dataclass(frozen=True)
class PromptDecision:
    action: str
    reason_code: str
    message: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "action": self.action,
            "reason_code": self.reason_code,
            "message": self.message,
        }


def load_security_config() -> tuple[dict, dict]:
    return (
        json.loads(DESCRIPTORS_PATH.read_text(encoding="utf-8")),
        json.loads(POLICY_PATH.read_text(encoding="utf-8")),
    )


DESCRIPTORS, POLICY = load_security_config()
TOOL_MAP = {tool["name"]: tool for tool in DESCRIPTORS["tools"]}
ALLOWED_TOOLS = frozenset(POLICY["tool_execution"]["allowlist"])
MENU_IDS = frozenset(item.item_id for item in load_menu())


INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(all|any|the|your)?\s*(previous|prior|system|developer)\s+(instructions?|prompts?|rules?)\b", re.I),
    re.compile(r"\b(reveal|print|repeat|show|leak|expose)\b.{0,40}\b(system prompt|hidden instructions?|api keys?|secrets?|environment variables?)\b", re.I),
    re.compile(r"\b(act as|pretend to be|enter)\b.{0,30}\b(developer mode|jailbreak|unrestricted mode)\b", re.I),
    re.compile(r"\b(call|invoke|use)\b.{0,30}\b(shell|terminal|filesystem|browser|unapproved tool)\b", re.I),
)


def contains_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in INJECTION_PATTERNS)


def precheck_user_request(text: str) -> PromptDecision:
    """Resolve deterministic safety and ambiguity cases before an AI call."""
    value = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not value:
        return PromptDecision("clarification", "empty_request", "Please describe what you want to eat today.")
    if contains_prompt_injection(value):
        return PromptDecision(
            "guardrail_blocked",
            "prompt_injection_detected",
            "Please restate the request using only food preferences or meal-planning information.",
        )

    restriction_patterns = (
        r"\bi will never eat\b",
        r"\bi (?:do not|don't|won't) want to eat anything\b",
        r"\bstop eating completely\b",
        r"\bstarv(?:e|ing)\b",
        r"\bzero calories\b",
        r"\bno food for the whole day\b",
    )
    extreme_target_patterns = (
        r"\b[4-9][0-9]{3,}\s*(?:kcal|calories)\b",
        r"\b[3-9][0-9]{2,}\s*(?:g|grams?)\s*(?:of )?protein\b",
    )
    restriction = any(re.search(pattern, value) for pattern in restriction_patterns)
    extreme_target = any(re.search(pattern, value) for pattern in extreme_target_patterns)
    if restriction and extreme_target:
        return PromptDecision(
            "clarification",
            "unsafe_conflicting_intake",
            "You requested both no food and a very high intake. Please confirm that you want a normal meal plan.",
        )
    if restriction:
        return PromptDecision(
            "safety_stop",
            "extreme_restriction",
            "MacroFit cannot create a starvation or extreme-restriction plan. Please request a normal meal plan.",
        )

    if any(re.search(pattern, value) for pattern in (
        r"\bthat meal\b",
        r"\bthe other meal\b",
        r"\bthe previous meal\b",
        r"\bthe same one\b",
    )):
        return PromptDecision(
            "clarification",
            "missing_meal_reference",
            "Please name breakfast, lunch or dinner instead of referring to 'that meal' or 'the other meal'.",
        )

    vegetarian = re.compile(r"\b(?:veg|vegetarian|meat-free|meatless)\b")
    chicken = re.compile(r"\b(?:chicken|non-veg|non-vegetarian|meat)\b")
    clauses = [part.strip() for part in re.split(r"[,;.]|\band\b", value) if part.strip()]
    for meal in ("breakfast", "lunch", "dinner"):
        meal_clauses = [clause for clause in clauses if re.search(rf"\b{meal}\b", clause)]
        has_vegetarian = any(vegetarian.search(clause) for clause in meal_clauses)
        has_chicken = any(chicken.search(clause) for clause in meal_clauses)
        if has_vegetarian and has_chicken:
            return PromptDecision(
                "clarification",
                "conflicting_dietary_requirement",
                f"You requested both vegetarian and chicken for {meal}. Which one should MacroFit use?",
            )
    return PromptDecision("continue", "passed")


def validate_untrusted_text(value: str, field: str) -> str:
    if any(ord(char) < 32 and char not in "\n\t" for char in value):
        raise GuardrailViolation("invalid_text", f"{field} contains unsupported control characters.")
    if contains_prompt_injection(value):
        raise GuardrailViolation(
            "prompt_injection_detected",
            f"Please restate {field} using only food preferences or meal-planning information.",
        )
    return value.strip()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _resolve_schema(schema: dict) -> dict:
    ref = schema.get("$ref")
    if not ref:
        return schema
    prefix = "#/$defs/"
    if not ref.startswith(prefix):
        raise GuardrailViolation("invalid_contract", "The tool contract contains an unsupported reference.")
    return DESCRIPTORS["$defs"][ref[len(prefix):]]


def _validate_schema(value: Any, schema: dict, path: str) -> None:
    schema = _resolve_schema(schema)
    expected = schema.get("type")
    valid_type = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": _is_number(value),
    }.get(expected, True)
    if not valid_type:
        raise GuardrailViolation("invalid_type", f"{path} must be {expected}.")

    if "enum" in schema and value not in schema["enum"]:
        raise GuardrailViolation("invalid_choice", f"{path} is not an allowed value.")

    if expected == "object":
        properties = schema.get("properties", {})
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            raise GuardrailViolation("missing_field", f"Missing required field: {sorted(missing)[0]}.")
        unknown = set(value) - set(properties)
        if schema.get("additionalProperties") is False and unknown:
            raise GuardrailViolation("unknown_field", f"Unknown field: {sorted(unknown)[0]}.")
        for key, child in value.items():
            if key in properties:
                _validate_schema(child, properties[key], f"{path}.{key}")
    elif expected == "array":
        if len(value) > schema.get("maxItems", float("inf")):
            raise GuardrailViolation("too_many_items", f"{path} has too many values.")
        for index, child in enumerate(value):
            _validate_schema(child, schema.get("items", {}), f"{path}[{index}]")
    elif expected == "string":
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", float("inf")):
            raise GuardrailViolation("invalid_length", f"{path} has an invalid length.")
        if schema.get("pattern") and not re.fullmatch(schema["pattern"], value):
            raise GuardrailViolation("invalid_format", f"{path} has an invalid format.")
    elif expected in {"number", "integer"}:
        if value < schema.get("minimum", -float("inf")) or value > schema.get("maximum", float("inf")):
            raise GuardrailViolation("out_of_range", f"{path} is outside the allowed range.")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise GuardrailViolation("out_of_range", f"{path} must be greater than {schema['exclusiveMinimum']}.")


def validate_tool_call(tool_name: str, arguments: dict) -> dict:
    """Validate a model-proposed tool call before any execution or state change."""
    if tool_name not in ALLOWED_TOOLS or tool_name not in TOOL_MAP:
        raise GuardrailViolation("tool_not_allowed", "That tool is not available to MacroFit.")
    if not isinstance(arguments, dict):
        raise GuardrailViolation("invalid_arguments", "Tool arguments must be a JSON object.")
    encoded = json.dumps(arguments, ensure_ascii=True).encode("utf-8")
    if len(encoded) > MAX_TOOL_ARGUMENT_BYTES:
        raise GuardrailViolation("payload_too_large", "The tool request is too large.")

    _validate_schema(arguments, TOOL_MAP[tool_name]["input_schema"], tool_name)

    free_text_fields = {"food_mood", "user_requested_food", "requested_food"}
    for key in free_text_fields & arguments.keys():
        arguments[key] = validate_untrusted_text(arguments[key], key)
    for key in ("dietary_restrictions", "allergies"):
        for index, text in enumerate(arguments.get(key, [])):
            arguments[key][index] = validate_untrusted_text(text, key)

    item_ids = []
    for key in ("item_id", "rejected_item_id"):
        if key in arguments:
            item_ids.append(arguments[key])
    item_ids.extend(entry.get("item_id") for entry in arguments.get("actual_items", []) if isinstance(entry, dict))
    item_ids.extend(arguments.get("excluded_item_ids", []))
    item_ids.extend(arguments.get("recent_item_ids", []))
    for item_id in item_ids:
        if item_id not in MENU_IDS:
            raise GuardrailViolation("unknown_item", f"{item_id} is not an approved dataset item.")
    return arguments


def validate_model_selection(selected_item_or_plan_id: str, allowed_ids: set[str] | frozenset[str]) -> str:
    """Stop the model from inventing an item or escaping the server shortlist."""
    if selected_item_or_plan_id not in allowed_ids:
        raise GuardrailViolation("selection_not_allowed", "The selection is outside the approved shortlist.")
    return selected_item_or_plan_id


def authorize_state_change(tool_name: str, user_confirmed: bool) -> None:
    if tool_name in {"log_actual_food"} and not user_confirmed:
        raise GuardrailViolation("confirmation_required", "Please confirm what was actually consumed before it is logged.")


def authorize_extra_food(projected_calories: float, target_calories: float, user_confirmed: bool) -> None:
    if projected_calories > target_calories and not user_confirmed:
        difference = round(projected_calories - target_calories)
        raise GuardrailViolation(
            "confirmation_required",
            f"This is approximately {difference} kcal above today's estimate. Confirm before logging it.",
        )


def safe_scope_response() -> str:
    return POLICY["assistant_scope"]["out_of_scope_response"]
