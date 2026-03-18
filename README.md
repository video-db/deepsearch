<!-- PROJECT SHIELDS -->
[![Python][python-shield]][python-url]
[![VideoDB][videodb-shield]][videodb-url]
[![LangGraph][langgraph-shield]][langgraph-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Website][website-shield]][website-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://videodb.io/">
    <img src="https://codaio.imgix.net/docs/_s5lUnUCIU/blobs/bl-RgjcFrrJjj/d3cbc44f8584ecd42f2a97d981a144dce6a66d83ddd5864f723b7808c7d1dfbc25034f2f25e1b2188e78f78f37bcb79d3c34ca937cbb08ca8b3da1526c29da9a897ab38eb39d084fd715028b7cc60eb595c68ecfa6fa0bb125ec2b09da65664a4f172c2f" alt="Logo" width="300" height="">
  </a>

  <h1 align="center">DeepSearch</h1>

  <p align="center">
    Stateful, multi-turn video retrieval with a production-grade indexing pipeline built on VideoDB.
    <br />
    <a href="https://docs.videodb.io"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="#quick-start">Quick Start</a>
    ·
    <a href="#features">Features</a>
    ·
    <a href="#how-it-works">How It Works</a>
    ·
    <a href="https://github.com/video-db/deepsearch/issues">Report Bug</a>
  </p>
</p>

---

## Why DeepSearch

Traditional video search often collapses complex intent into one embedding query.

DeepSearch improves relevance by combining:
- **Indexing orchestration** (scene extraction, transcript, object detection, multimodal enrichment, summary indexes)
- **Retrieval orchestration** (LLM-planned multi-index search, validator loop, reranking, and follow-up refinement)
- **Stateful retrieval memory** (context-aware refinement across conversational turns)

---

## Features

- Index from a public video URL or an existing VideoDB media ID
- Structured indexing telemetry via event callbacks for progress visibility
- Multi-turn search sessions with `search`, `followup`, `resume_session`
- Explainable clip results (primary subquery, primary index, supporting subqueries)
- Robust conversational continuity with persisted session state across turns
- Pluggable stores with SQLite defaults for sessions, index records, metadata, and artifacts
- Configurable model routing and per-stage model overrides
- Vision-aware enrichment for stronger multimodal retrieval quality

---

## How It Works

DeepSearch has two connected runtimes:
- **Indexing runtime** builds semantic indexes from scenes, transcript, and optional object signals.
- **Retrieval runtime** runs a stateful LangGraph loop that plans queries, validates candidates, reranks clips, and supports follow-up turns.

For each user query, DeepSearch returns ranked clips with explainability fields so you can see why each clip matched.

---

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) 0.8+
- VideoDB API key ([console.videodb.io](https://console.videodb.io))
- OpenAI-compatible API key for configured LLM route

### 1) Install

Recommended install (best retrieval quality):

```bash
uv sync --extra detection
```

DeepSearch uses object detection during indexing to add object-level visual signals (for example: person, laptop, car, traffic sign) into scene metadata. Retrieval then uses those signals during ranking and refinement, which improves results for object-centric queries.

If you want a lighter setup without local detector dependencies, you can still run DeepSearch by installing base deps only and disabling detection in config.

Base-only install:

```bash
uv sync
```

Then set `indexing.object_detection.mode` to a non-local value in `deepsearch_config.yaml` to skip detection:

```yaml
indexing:
  object_detection:
    mode: off
```

### 2) Configure env

```bash
cp .env.sample .env
```

Set at minimum:
- `VIDEO_DB_API_KEY`
- `OPENAI_API_KEY`

Optional:
- `DEEPSEARCH_DB_PATH`
- `DEEPSEARCH_CONFIG` (defaults to `deepsearch_config.yaml`)

### 3) Index a video

`--collection-id` is optional. If you already have a VideoDB collection, pass its ID. If you leave it empty, DeepSearch falls back to your account default collection via the SDK.

```bash
uv run python index_video.py \
  [--collection-id <collection_id>] \
  --video-url <public_video_url>
```

Or index an existing VideoDB media object:

```bash
uv run python index_video.py \
  [--collection-id <collection_id>] \
  --media-id <media_id>
```

If your source video is local, upload it to VideoDB first, copy the returned `media_id`, then run indexing with `--media-id`.

```python
import videodb

conn = videodb.connect(api_key="YOUR_VIDEO_DB_API_KEY")
collection = conn.get_collection()  # or conn.get_collection("<collection_id>")

# Upload local media file
video = collection.upload(file_path="./videos/my_video.mp4", name="My Local Video")

print("media_id:", video.id)
```

Then index it with DeepSearch:

```bash
uv run python index_video.py \
  [--collection-id <collection_id>] \
  --media-id <media_id>
```

### 4) Run interactive retrieval

```bash
uv run python run_deepsearch.py \
  [--collection-id <collection_id>] \
  --query "rainy night scenes with emotional dialogue"
```

Interactive commands:
- `/more` for next page
- `/help` for command help
- `/exit` to end and print `session_id`

---

## Configuration

DeepSearch supports typed config, dict config, and YAML-file config.

- Default config file: `deepsearch_config.yaml`
- Config schema: `deepsearch/config/schema.py`
- Environment overrides via `DeepSearchConfig.from_env()` with `DEEPSEARCH_` prefix and nested keys using double underscores

Example:

```bash
export DEEPSEARCH_RETRIEVAL__PAGE_SIZE=20
```

---

## Project Structure

```text
deepsearch/
├── client.py                     # Public client/session entrypoints
├── indexing/                     # Indexing pipeline + stage contracts
├── retrieval/                    # LangGraph retrieval graph + nodes
├── providers/                    # LLM and detector provider adapters
├── stores/                       # Session/metadata/index record stores
├── config/                       # Typed config schema and defaults
├── telemetry/                    # Logging utilities
└── errors/                       # Error taxonomy and typed errors

index_video.py                    # CLI script for indexing
run_deepsearch.py                 # Interactive retrieval script
sample_end_user_usage.py          # End-user API walkthrough
deepsearch_config.yaml            # Example config
docs/PRD.md                       # Product requirements draft
docs/specs.md                     # Technical specs draft
```

---

## Troubleshooting

### Missing API key

If initialization fails with `VIDEO_DB_API_KEY is required`, verify `.env` is loaded and the key is present.

### Local detector import errors

If detection stage raises missing modules (`torch`, `transformers`, etc.), either install detection extras or disable local detection mode in config.

```bash
uv sync --extra detection
```

### Resume an indexing run

If indexing fails after upload/extract, rerun `index_video` with the printed `media_id` (and optionally `--force-reindex`) to continue from persisted artifacts.

---

## Community & Support

- **Docs**: [docs.videodb.io](https://docs.videodb.io)
- **Issues**: [GitHub Issues](https://github.com/video-db/deepsearch/issues)
- **Discord**: [Join community](https://discord.gg/py9P639jGz)
- **Console**: [Get API key](https://console.videodb.io)

---

<p align="center">Made with ❤️ by the <a href="https://videodb.io">VideoDB</a> team</p>

---

<!-- MARKDOWN LINKS & IMAGES -->
[python-shield]: https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[videodb-shield]: https://img.shields.io/badge/VideoDB-SDK-0EA5A4?style=for-the-badge
[videodb-url]: https://videodb.io/
[langgraph-shield]: https://img.shields.io/badge/LangGraph-Orchestration-111827?style=for-the-badge
[langgraph-url]: https://www.langchain.com/langgraph
[stars-shield]: https://img.shields.io/github/stars/video-db/deepsearch.svg?style=for-the-badge
[stars-url]: https://github.com/video-db/deepsearch/stargazers
[issues-shield]: https://img.shields.io/github/issues/video-db/deepsearch.svg?style=for-the-badge
[issues-url]: https://github.com/video-db/deepsearch/issues
[website-shield]: https://img.shields.io/website?url=https%3A%2F%2Fvideodb.io%2F&style=for-the-badge&label=videodb.io
[website-url]: https://videodb.io/
