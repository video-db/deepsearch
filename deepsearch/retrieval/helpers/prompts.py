from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

import yaml

from deepsearch.retrieval.helpers import registry, reason_codes

STRICT_JSON_RULES = """Rules:
- Output STRICT JSON only. No prose, no markdown, no comments.
- Use double quotes for all keys/strings. No trailing commas. No extra fields.
"""

VALIDATOR_ALLOWED_OPS = [
    "drop_filter",
    "drop_values",
    "set_subquery",
    "by_example",
    "merge_indexes",
]
INTERPRETER_ALLOWED_OPS = [
    "merge_filter",
    "drop_filter",
    "drop_values",
    "set_subquery",
    "by_example",
    "merge_indexes",
]
NONE_ALLOWED_OPS = [
    "drop_filter",
    "drop_values",
    "set_subquery",
    "by_example",
    "merge_indexes",
    "set_join_plan",
]

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PROMPT_FILE = _PROMPT_DIR / "deepsearch_run.yaml"
logger = logging.getLogger(__name__)


def _load_prompt(name: str) -> str:
    if _PROMPT_FILE.exists():
        prompts = yaml.safe_load(_PROMPT_FILE.read_text(encoding="utf-8")) or {}
        if name in prompts:
            return str(prompts[name])
    path = _PROMPT_DIR / f"{name}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt template not found: {_PROMPT_FILE} or {path}")


class PromptFactory:
    def __init__(self, unique_metadata: dict, debug_mode: bool = False):
        self.unique_metadata = unique_metadata
        self.debug_mode = debug_mode
        if self.debug_mode:
            logger.debug("PromptFactory unique metadata: %s", self.unique_metadata)

    def _dbg_prompt(self, name: str, prompt: str) -> str:
        unresolved = re.findall(r"\{\{[^{}]+\}\}", prompt)
        if unresolved:
            logger.warning(
                "Prompt[%s] has unresolved placeholders: %s",
                name,
                sorted(set(unresolved)),
            )
        if self.debug_mode:
            logger.debug("Prompt[%s]:\n%s", name, prompt)
        return prompt

    def _facet_csv(self) -> str:
        return ", ".join(registry.allowed_facet_names())

    def _per_value_csv(self) -> str:
        pv = registry.per_value_relax_facets()
        return ", ".join(pv) if pv else "(none)"

    def build_plan_init_prompt(self, user_query: str) -> str:
        tmpl = _load_prompt("plan_init")
        return self._dbg_prompt(
            "plan_init",
            (
                tmpl.replace("{{facet_csv}}", self._facet_csv())
                .replace("{{registry_fallback_order}}", str(registry.fallback_order()))
                .replace("{{registry_indexes_doc}}", registry.indexes_doc())
                .replace("{{registry_facets_doc}}", registry.facets_doc())
                .replace(
                    "{{unique_metadata_emotions}}",
                    str(self.unique_metadata.get("emotions", [])),
                )
                .replace(
                    "{{unique_metadata_shot_types}}",
                    str(self.unique_metadata.get("shot_types", [])),
                )
                .replace(
                    "{{unique_metadata_relationships}}",
                    str(self.unique_metadata.get("relationships", [])),
                )
                .replace("{{strict_json_rules}}", STRICT_JSON_RULES)
                .replace("{{reason_codes_PLAN_INIT}}", reason_codes.PLAN_INIT)
                .replace("{{user_query}}", user_query)
            ),
        )

    def build_paraphrase_prompt(self, query: str, k: int, main_query: str) -> str:
        tmpl = _load_prompt("paraphrase")
        return self._dbg_prompt(
            "paraphrase",
            (
                tmpl.replace("{{num_paraphrases}}", str(k))
                .replace("{{query}}", query)
                .replace("{{main_query}}", main_query)
                .replace("{{strict_json_rules}}", STRICT_JSON_RULES)
            ),
        )

    def build_validator_prompt(self, main_query: str) -> str:
        tmpl = _load_prompt("validator")
        return self._dbg_prompt(
            "validator",
            (
                tmpl.replace("{{main_query}}", main_query)
                .replace("{{facet_csv}}", self._facet_csv())
                .replace(
                    "{{operations_doc_for_validator}}",
                    registry.operations_doc_for(VALIDATOR_ALLOWED_OPS, with_usage=True),
                )
                .replace("{{strict_json_rules}}", STRICT_JSON_RULES)
                .replace(
                    "{{reason_codes_validator_summary}}", reason_codes.VALIDATOR_SUMMARY
                )
                .replace(
                    "{{reason_codes_validator_feedback}}",
                    reason_codes.VALIDATOR_FEEDBACK,
                )
            ),
        )

    def build_interpreter_prompt(self) -> str:
        tmpl = _load_prompt("interpreter")
        return self._dbg_prompt(
            "interpreter",
            (
                tmpl.replace("{{facet_csv}}", self._facet_csv())
                .replace(
                    "{{operations_doc_for_interpreter}}",
                    registry.operations_doc_for(
                        INTERPRETER_ALLOWED_OPS, with_usage=True
                    ),
                )
                .replace(
                    "{{reason_codes_BATCH_FROM_USER}}", reason_codes.BATCH_FROM_USER
                )
                .replace("{{reason_codes_INTENT_ROUTE}}", reason_codes.INTENT_ROUTE)
            ),
        )

    def build_none_prompt(self) -> str:
        tmpl = _load_prompt("none_analyzer")
        return self._dbg_prompt(
            "none_analyzer",
            (
                tmpl.replace("{{facet_csv}}", self._facet_csv())
                .replace("{{per_value_csv}}", self._per_value_csv())
                .replace(
                    "{{operations_doc_for_none}}",
                    registry.operations_doc_for(NONE_ALLOWED_OPS, with_usage=True),
                )
                .replace("{{strict_json_rules}}", STRICT_JSON_RULES)
                .replace(
                    "{{reason_codes_DELTA_FROM_NONE}}", reason_codes.DELTA_FROM_NONE
                )
            ),
        )

    def build_reranker_prompt(self) -> str:
        tmpl = _load_prompt("reranker")
        return self._dbg_prompt(
            "reranker",
            tmpl.replace("{{strict_json_rules}}", STRICT_JSON_RULES).replace(
                "{{reason_codes_RERANK_PREF}}", reason_codes.RERANK_PREF
            ),
        )

    def build_create_query_from_description_prompt(self) -> str:
        tmpl = _load_prompt("create_query_from_description")
        return self._dbg_prompt(
            "create_query_from_description",
            tmpl.replace("{{strict_json_rules}}", STRICT_JSON_RULES),
        )

    def build_object_names_prompt(self, query_object: list[str]) -> str:
        tmpl = _load_prompt("objects")
        return self._dbg_prompt(
            "objects",
            tmpl.replace("{{query_object}}", str(query_object)).replace(
                "{{unique_metadata_objects}}",
                str(self.unique_metadata.get("objects", [])),
            ),
        )
