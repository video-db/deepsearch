#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from deepsearch import DeepSearchClient

load_dotenv()


def _looks_like_media_id(value: str | None) -> bool:
    if not value:
        return False
    return value.startswith("m-")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a video with DeepSearch")
    parser.add_argument(
        "--config", default=os.getenv("DEEPSEARCH_CONFIG", "deepsearch_config.yaml")
    )
    parser.add_argument("--collection-id", default="")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video-url")
    source_group.add_argument("--media-id")
    parser.add_argument("--name")
    parser.add_argument("--api-key", default=os.getenv("VIDEO_DB_API_KEY"))
    parser.add_argument(
        "--base-url", default=os.getenv("VIDEO_DB_BASE_URL", "https://api.videodb.io")
    )
    parser.add_argument("--force-reindex", action="store_true")
    args = parser.parse_args()

    if args.video_url and _looks_like_media_id(args.video_url):
        raise SystemExit(
            "It looks like you passed a VideoDB media ID to --video-url. "
            "Use --media-id <m-...> instead."
        )

    config = args.config if os.path.exists(args.config) else None
    client = DeepSearchClient(
        config=config, api_key=args.api_key, base_url=args.base_url
    )
    manifest = client.index_video(
        collection_id=args.collection_id,
        video_url=args.video_url,
        media_id=args.media_id,
        name=args.name,
        on_event=lambda e: print(f"[{e.stage}] {e.status} - {e.message or ''}"),
        force_reindex=args.force_reindex,
    )

    print("manifest_id:", manifest.manifest_id)
    print("video_id:", manifest.video_id)
    print("indexes:", ", ".join(sorted(manifest.indexes)))


if __name__ == "__main__":
    main()
