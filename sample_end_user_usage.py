#!/usr/bin/env python3
"""
Illustrative DeepSearch end-user script (real API shape).

This script shows how v0.1 is expected to look for users.
It assumes DeepSearch is installed and configured.
"""

from __future__ import annotations

import importlib
from pprint import pprint


def print_result(title, result):
    print(f"\n=== {title} ===")
    print("session_id:", result.session_id)
    print("waiting_for:", result.waiting_for)
    print("page:", result.page)
    for clip in result.clips:
        print(
            f"- rank={clip.rank} [{clip.start:.1f}s-{clip.end:.1f}s] "
            f"primary_index={clip.explain.primary_index}"
        )


def on_index_event(event):
    print(f"[index:{event.stage}] {event.status} progress={event.progress}")


def main():
    DeepSearchClient = getattr(importlib.import_module("deepsearch"), "DeepSearchClient")

    cfg_mod = importlib.import_module("deepsearch.config")
    DeepSearchConfig = getattr(cfg_mod, "DeepSearchConfig")
    LLMConfig = getattr(cfg_mod, "LLMConfig")
    LLMModelsConfig = getattr(cfg_mod, "LLMModelsConfig")
    LLMIndexingModelsConfig = getattr(cfg_mod, "LLMIndexingModelsConfig")
    LLMRetrievalModelsConfig = getattr(cfg_mod, "LLMRetrievalModelsConfig")
    RetrievalConfig = getattr(cfg_mod, "RetrievalConfig")

    config = DeepSearchConfig(
        llm=LLMConfig(
            route="vercel_ai_sdk_python",
            provider_mode="direct",
            models=LLMModelsConfig(
                indexing=LLMIndexingModelsConfig(
                    scene_enrichment="o3",
                    subplot_summary="o3",
                    final_summary="o3",
                ),
                retrieval=LLMRetrievalModelsConfig(
                    planner="gpt-4o-2024-11-20",
                    paraphrase="gpt-4o-2024-11-20",
                    validator="gpt-4o-2024-11-20",
                    none_analyzer="gpt-4o-2024-11-20",
                    interpreter="gpt-4o-2024-11-20",
                    reranker="gpt-4o-2024-11-20",
                ),
            ),
        ),
        retrieval=RetrievalConfig(page_size=2),
    )

    client = DeepSearchClient(
        config=config
    )

    # Exactly one is required: `video_url` or `media_id`
    manifest = client.index_video(
        collection_id="c_demo_001",
        video_url="https://example.com/demo_video.mp4",
        on_event=on_index_event,
    )

    print("\n=== Index Manifest ===")
    pprint(
        {
            "manifest_id": manifest.manifest_id,
            "video_id": manifest.video_id,
            "indexes": list(manifest.indexes.keys()),
        }
    )

    session = client.start_session(collection_id="c_demo_001", page_size=2)

    result1 = session.search(
        "rainy night scenes with close framing and emotional dialogue"
    )
    print_result("Search Page 1", result1)

    # `show_more` is pagination only in v0.1 (no fresh retrieval run)
    result2 = session.followup(ui_event={"type": "show_more", "payload": {}})
    print_result("Search Page 2", result2)

    # Refinement follow-up updates plan and reruns retrieval
    result3 = session.followup(
        text="keep only high emotion and remove wide shots",
        page_size=2,
    )
    print_result("Refined Results", result3)


if __name__ == "__main__":
    main()
