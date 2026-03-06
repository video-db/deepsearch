from __future__ import annotations

from typing import Dict, List, Union

INDEXES: Dict[str, Dict] = {
    "location": {
        "label": "The setting/environment is the focus (forest, marketplace, rooftop, rain, sunset cityscape)."
    },
    "action": {
        "label": "Physical actions, movements, and interactions (running, fighting, driving, picking up an object, opening a door)."
    },
    "scene_description": {
        "label": "Broad visual description of what's on screen (costumes, colors, ambience, staging)."
    },
    "transcript": {
        "label": "The query quotes or directly references exact words/lyrics."
    },
    "topic": {
        "label": "The query describes what is being discussed/sung. Trigger when query contains words like talking, calling, discussing, singing, etc."
    },
    "object_description": {
        "label": "Use when query centers on an object's attributes/state/identity. Include color, material, condition, or distinctive parts."
    },
    "subplot_summary": {
        "label": "Use when the query narrates a story arc spanning a segment or subplot."
    },
    "final_summary": {
        "label": "Use when the query clearly spans the entire film from setup to resolution."
    },
}

FACETS: Dict[str, Dict] = {
    "shot_type": {
        "label": "Shot Type",
        "per_value_relax": False,
        "drop_supported": True,
        "fallback_priority": 3,
    },
    "emotion": {
        "label": "Emotion",
        "per_value_relax": False,
        "drop_supported": True,
        "fallback_priority": 2,
    },
    "objects": {
        "label": "Objects",
        "per_value_relax": False,
        "drop_supported": True,
        "fallback_priority": 1,
    },
}

OPERATIONS: Dict[str, str] = {
    "set_filter": '{{"op":"set_filter","facet":"<one of: {FACETS}>","values":["value1","value2"]}}',
    "merge_filter": '{{"op":"merge_filter","facet":"<one of: {FACETS}>","values":["value1","value2"]}}',
    "drop_filter": '{{"op":"drop_filter","facet":"<one of: {FACETS}>}}',
    "drop_values": '{{"op":"drop_values","facet":"<one of: {PER_VALUE_FACETS}>","names":["value1","value2"]}}',
    "set_subquery": '{{"op":"set_subquery","subquery_id":"Qx","q":"..."}}',
    "by_example": '{{"op":"by_example","video_id":"...","start":0.0,"end":5.0,"text":"..."}}',
    "merge_indexes": '{{"op":"merge_indexes","subquery_id":"Qx","indexes_to_add":["__"]}}',
    "set_join_plan": '{{"op":"set_join_plan","join_plan":{{"op":"AND|OR","subqueries":["Q1","Q2"]}}}}',
}

OP_USAGE: Dict[str, str] = {
    "set_filter": "Replace a facet entirely with new values (tighten/override).",
    "merge_filter": "Add values to a facet without removing existing (broaden recall).",
    "drop_filter": "Remove a facet constraint altogether (maximize recall on that facet).",
    "drop_values": "Remove specific values from a per-value facet (keep the rest).",
    "set_subquery": "Retain structure but tweak the wording of an existing subquery.",
    "by_example": "Seed retrieval from a shown clip when user says 'more like #N'.",
    "merge_indexes": "Add indexes to the same subquery to boost recall (no intent shift).",
    "set_join_plan": "Switch boolean join (AND/OR) when the current join is too strict/loose.",
}


def operations_doc_for(ops: List[str], with_usage: bool = False) -> str:
    facet_csv = ", ".join(allowed_facet_names())
    per_value_csv = ", ".join(per_value_relax_facets()) or "(none)"
    lines: List[str] = []
    for name in ops:
        tmpl = OPERATIONS.get(name)
        if not tmpl:
            continue
        rendered = tmpl.format(FACETS=facet_csv, PER_VALUE_FACETS=per_value_csv)
        if with_usage:
            lines.append(f"- {rendered} — When: {OP_USAGE.get(name, '')}")
        else:
            lines.append(f"- {rendered}")
    return "\n    ".join(lines)


def allowed_index_names() -> List[str]:
    return list(INDEXES.keys())


def allowed_facet_names() -> List[str]:
    return list(FACETS.keys())


def is_valid_index(name: Union[str, List[str]]) -> bool:
    if isinstance(name, str):
        return name in INDEXES
    if isinstance(name, list):
        return all(isinstance(n, str) and n in INDEXES for n in name)
    return False


def is_valid_facet(name: Union[str, List[str]]) -> bool:
    if isinstance(name, str):
        return name in FACETS
    if isinstance(name, list):
        return all(isinstance(n, str) and n in FACETS for n in name)
    return False


def per_value_relax_facets() -> List[str]:
    return [k for k, v in FACETS.items() if v.get("per_value_relax")]


def fallback_order() -> List[str]:
    return [
        k
        for k, _ in sorted(
            FACETS.items(), key=lambda kv: kv[1].get("fallback_priority", 99)
        )
    ]


def indexes_doc() -> str:
    lines = ["Allowed indexes:"]
    for k, v in INDEXES.items():
        lines.append(f"- {k} — {v.get('label', '')}")
    return "\n".join(lines)


def facets_doc() -> str:
    lines = ["Allowed metadata facets (all values are lists of strings):"]
    for k, v in FACETS.items():
        flags = []
        if v.get("per_value_relax"):
            flags.append("per-value-relax")
        if v.get("drop_supported"):
            flags.append("drop-supported")
        lines.append(
            f"- {k} — {v.get('label', '')} ({', '.join(flags) if flags else 'no special flags'})"
        )
    return "\n".join(lines)
