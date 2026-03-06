from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deepsearch.retrieval.state import GraphState
from deepsearch.retrieval.nodes.plan_init import plan_init_node
from deepsearch.retrieval.nodes.search_join import search_join_node
from deepsearch.retrieval.nodes.validator import validator_node
from deepsearch.retrieval.nodes.none_analyzer import none_analyzer_node
from deepsearch.retrieval.nodes.interpreter import interpreter_node
from deepsearch.retrieval.nodes.reranker import rerank_node


def preview_node(state: GraphState):
    return Command(update={"paused_for": "preview_pause"}, goto=END)


def clarify_node(state: GraphState):
    return Command(update={"paused_for": "clarify_pause"}, goto=END)


def _routing_function(state: GraphState):
    if state.paused_for:
        return "interpreter"
    return "plan_init"


def build_graph() -> object:
    graph = StateGraph(GraphState)

    for name, fn in {
        "plan_init": plan_init_node,
        "search_join": search_join_node,
        "validator": validator_node,
        "none_analyzer": none_analyzer_node,
        "interpreter": interpreter_node,
        "rerank": rerank_node,
        "preview_pause": preview_node,
        "clarify_pause": clarify_node,
    }.items():
        graph.add_node(name, fn)

    graph.add_conditional_edges(
        START,
        _routing_function,
        {"interpreter": "interpreter", "plan_init": "plan_init"},
    )
    return graph.compile()
