#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from deepsearch import DeepSearchClient
from deepsearch.retrieval.contracts import UiEvent
from deepsearch.stores.sqlite import SQLiteSessionStore

load_dotenv()


def print_result(result) -> None:
    print("session_id:", result.session_id)
    print("waiting_for:", result.waiting_for)
    if result.waiting_for == "clarification":
        clarification = result.clarification
        prompt = (
            clarification.text.strip()
            if clarification and clarification.text
            else "Could you clarify your request?"
        )
        print("clarification:", prompt)
        if clarification and clarification.mode == "mcq" and clarification.options:
            for idx, option in enumerate(clarification.options):
                letter = chr(ord("A") + idx) if idx < 26 else str(idx + 1)
                print(f"  {letter}. {option.label}")
            print("reply_format: option or natural language")
        else:
            print("reply_format: natural language")
        if result.debug:
            print("debug:", result.debug)
        return
    print("page:", result.page.model_dump())
    for clip in result.clips:
        print(
            f"- rank={clip.rank} video_id={clip.video_id} start={clip.start} end={clip.end} "
            f"score={clip.score} primary_index={clip.explain.primary_index} stream_url={clip.stream_url}"
        )
    if result.debug:
        print("debug:", result.debug)


def print_help() -> None:
    print("Commands:")
    print("  /more       Show next page from cached results")
    print("  /help       Show this help")
    print("  /exit       Exit and print session_id")
    print("  <text>      Send follow-up text")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DeepSearch retrieval interactive chat"
    )
    parser.add_argument(
        "--config",
        default=os.getenv("DEEPSEARCH_CONFIG", "deepsearch_config.yaml"),
    )
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--query")
    parser.add_argument("--video-id")
    parser.add_argument("--session-id")
    parser.add_argument("--page-size", type=int, default=5)
    parser.add_argument("--api-key", default=os.getenv("VIDEO_DB_API_KEY"))
    parser.add_argument(
        "--base-url", default=os.getenv("VIDEO_DB_BASE_URL", "https://api.videodb.io")
    )
    args = parser.parse_args()

    config = args.config if os.path.exists(args.config) else None
    session_store = SQLiteSessionStore(os.getenv("DEEPSEARCH_DB_PATH"))
    client = DeepSearchClient(
        config=config,
        api_key=args.api_key,
        base_url=args.base_url,
        session_store=session_store,
    )

    if args.session_id:
        session = client.resume_session(args.session_id)
        print(f"Resumed session_id: {session.session_id}")
        result = session.result
        print("current_result:")
        print_result(result)
    else:
        session = client.start_session(
            collection_id=args.collection_id,
            video_id=args.video_id,
            page_size=args.page_size,
        )
        print(f"Started session_id: {session.session_id}")
        if args.query:
            result = session.search(args.query)
            print("first_result:")
            print_result(result)
        else:
            print("No initial query provided. Type a query to begin.")

    print_help()
    try:
        while True:
            user = input("deepsearch> ").strip()
            if not user:
                continue
            if user == "/help":
                print_help()
                continue
            if user == "/exit":
                print(f"Session ended. session_id={session.session_id}")
                break
            if user == "/more":
                result = session.followup(ui_event=UiEvent(type="show_more"))
                print_result(result)
                continue

            # default: text follow-up; if no prior search happened this acts as first search
            try:
                has_state = (
                    session._persisted is not None
                    or session._client._load_session(session.session_id) is not None
                )
            except Exception:
                has_state = False
            if has_state:
                result = session.followup(text=user)
            else:
                result = session.search(user)
            print_result(result)
    except (EOFError, KeyboardInterrupt):
        print(f"\nSession interrupted. session_id={session.session_id}")


if __name__ == "__main__":
    main()
