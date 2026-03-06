# DeepSearch PRD

Version: 0.1  
Status: Brainstorm Draft  
Date: 2026-03-05

## 1) Product Summary

DeepSearch is a Python library that provides an end-to-end workflow for:

1. Indexing videos into structured, searchable semantic representations.
2. Running iterative, conversational retrieval over those indexes.

DeepSearch uses VideoDB for extraction, storage, and search execution primitives. DeepSearch owns orchestration, enrichment, planning, validation, refinement loops, and session continuity.

## 2) Why This Product

Video search quality drops quickly when it is treated as one query over one embedding space. Real user intent mixes visual context, dialogue, objects, emotion, and story-level arcs.

DeepSearch solves this by:

- Building multiple semantic indexes per video (instead of one generic index).
- Decomposing a user query into a structured plan across indexes + filters.
- Validating and auto-correcting results in-loop.
- Supporting follow-up refinement without resetting context.

## 3) Goals

- Provide one cohesive DeepSearch library for both indexing and retrieval.
- Make model/provider selection configurable per stage.
- Default LLM routing through Vercel AI SDK Python.
- Support OpenRouter as an optional route.
- Keep architecture flexible for future third-party model platform adapters.
- Use LangGraph as the retrieval orchestration dependency.
- Preserve stateful conversational retrieval with robust pause/resume behavior.
- Return explainable results (why each clip matched).

## 4) Non-Goals (v0.1)

- Building a dedicated user-facing web app.
- Replacing VideoDB media infrastructure.
- Locking DeepSearch to one provider or one inference vendor.

## 5) Users

- AI application developers embedding deep video retrieval in Python backends.
- Product engineers building search + refinement experiences.
- Relevance engineers tuning index quality and retrieval behavior.

## 6) Product Scope

### 6.1 Indexing Scope

DeepSearch indexing pipeline must:

- Ingest from `video_url` or `media_id`.
- Use VideoDB extraction primitives for scene and transcript inputs.
- Run configurable enrichment models over multimodal inputs.
- Produce DeepSearch-compatible named indexes.
- Attach metadata used by retrieval facets.

### 6.2 Retrieval Scope

DeepSearch retrieval runtime must:

- Orchestrate search flow through a LangGraph state graph.
- Build/maintain a structured search plan.
- Run multi-index retrieval and join logic.
- Validate results and auto-recover when quality is poor or empty.
- Support follow-up turns in the same session.
- Return ranked clips with explainability payload.

## 7) Current Baseline (Grounded in Existing System)

- Retrieval orchestration is state-machine based with node-level transitions and pause/resume.
- Query-time retrieval uses named scene indexes and metadata filters.
- Indexing currently uses scene extraction + transcript + object detection + VLM extraction + scene index creation.
- Existing default models are strong enough to seed v0.1 defaults:
  - VLM extraction/summarization: `o3`
  - Retrieval reasoning baseline: `gpt-4o-2024-11-20`
  - Transcript indexing engine: `gemini` via spoken-word indexing flow
  - Object detection baseline: RT-DETR v2 local/compute path

## 8) Functional Requirements

### 8.1 Indexing Requirements

- Must generate and maintain the following index names:
  - `location`
  - `scene_description`
  - `transcript`
  - `topic`
  - `object_description`
  - `subplot_summary`
  - `final_summary`

- Must persist metadata fields needed for retrieval filters:
  - `shot_type`
  - `emotion`
  - `objects`

- Must allow per-stage model configuration (no hard-coded single model path).
- Must support local object detection with configurable backend and params.
- Must expose pipeline progress events and final index manifest programmatically.
- Re-index default policy: replace existing same-name indexes for the same video to keep one canonical latest set.

### 8.2 Retrieval Requirements

- Must preserve iterative loop behavior:
  - plan initialization
  - search + join
  - validator
  - none-analyzer recovery
  - interpreter-driven follow-up edits
  - rerank
  - pause on preview or clarification

- Must support follow-up intent handling:
  - show more
  - tighten/broaden constraints
  - by-example refinement
  - filter changes and subquery edits

- `show_more` contract in v0.1: paginate existing ranked results only; if exhausted, return no-more-results state without re-query.

- Must include bounded safety limits:
  - recursion cap
  - retry caps
  - max operation batch caps

### 8.3 Session/State Requirements

- Must support persisted session state keyed by `session_id`.
- Must allow resume with new user input and existing plan/history.
- Must keep state schema explicit and serializable.
- Must use a pluggable session-store interface.
- Default state storage is in-memory; SQLite is an adapter option, not a locked requirement.

### 8.4 Explainability Requirements

- Each returned clip should include:
  - primary matched subquery
  - primary index used
  - support subqueries
  - rank/score metadata
  - validator status where relevant

### 8.5 Flexibility Requirements

- LLM routing defaults to Vercel AI SDK Python.
- OpenRouter must be selectable through config.
- Provider interface must be extensible for future inference platforms without API breakage.

## 9) Python API (v0.1)

DeepSearch is exposed as a Python package. Proposed core surface:

```python
from deepsearch import DeepSearchClient

client = DeepSearchClient(config=...)

# Indexing (default: blocking/local)
manifest = client.index_video(
    video_url="...",
    # alternatively: media_id="m-...",
    collection_id="...",
    on_event=lambda e: print(e.stage, e.status),
)

# Retrieval
session = client.start_session(collection_id="...")
result1 = session.search("scenes where someone runs in heavy rain")
result2 = session.followup("only close-up shots")
```

### Required API Capabilities

- Single indexing API (`index_video`) that accepts either URL input or existing media ID.
- Blocking indexing path that runs fully on the developer machine.
- Optional progress callback/events for stage-level indexing visibility.
- Sessioned retrieval (`start_session`, `search`, `followup`, `resume_session`).

### 9.1 Execution Model

- DeepSearch v0.1 is library-local, not a hosted orchestration service.
- Indexing executes in-process by default.
- No server-side status API is required for core functionality.
- SQLite may optionally store stage progress artifacts for local workflows.
- `index_video` input contract: exactly one of `video_url` or `media_id` is required.

## 10) Configuration Model

Configuration must be declarative, overridable, and provider-agnostic.

DeepSearch should provide a typed config API for easier usage (instead of requiring full raw JSON), with helper constructors such as `defaults()`, `from_file(...)`, and `from_env(...)`.

### 10.1 LLM Routing

- `llm.route`: default `vercel_ai_sdk_python`
- `llm.provider_mode`: e.g. `direct` or `openrouter`
- task-level model mapping for indexing and retrieval

### 10.2 Suggested Defaults (v0.1)

These defaults are based on currently used working values and should be treated as initial baseline, not permanent constraints.

```yaml
llm:
  route: vercel_ai_sdk_python
  provider_mode: direct
  openrouter:
    enabled: false
    api_key_env: OPENROUTER_API_KEY

  models:
    indexing:
      scene_enrichment: o3
      subplot_summary: o3
      final_summary: o3
    retrieval:
      planner: gpt-4o-2024-11-20
      paraphrase: gpt-4o-2024-11-20
      validator: gpt-4o-2024-11-20
      none_analyzer: gpt-4o-2024-11-20
      interpreter: gpt-4o-2024-11-20
      reranker: gpt-4o-2024-11-20

indexing:
  scene_extraction:
    shot_based:
      threshold: 30
      frame_count: 10
    time_based:
      time: 1
      frame_count: 10

  transcript:
    method: index_spoken_words
    engine: gemini
    language_code: ""

  object_detection:
    mode: local
    backend: rtdetr_v2
    threshold: 0.85
    batch_size: 64

  vlm:
    llm_max_images: 10
    batch_size: 500
    generate_subplot: true
    subplot_chunk_size: 250
    retry_attempts: 3
    retry_backoff_sec: 1

retrieval:
  k_variants_per_index: 2
  topk_per_variant: 30
  validator_max: 40
  validator_batch_size: 8
  max_ops_per_batch: 4
  page_size: 10
  validator_max_tries: 3
  validator_max_resets: 3
  none_max_tries: 3
  none_max_resets: 3
  recursion_limit: 12
```

## 11) Indexing Pipeline Specification

1. Resolve input (`video_url` upload or existing `media_id`).
2. Trigger scene extraction (shot + time based).
3. Trigger spoken-word indexing/transcript fetch path.
4. Run object detection for frame-level object signals.
5. Run multimodal scene enrichment (frames + transcript + objects).
6. Optionally generate subplot/final summaries.
7. Materialize named scene indexes in VideoDB through one deterministic writer path.
8. Return/persist index manifest and stage diagnostics.

Index manifest must include all required index names; missing required indexes should fail with a structured missing-index error.

## 12) Retrieval Loop Specification

1. Build initial plan from user query and collection metadata.
2. Generate paraphrases per subquery.
3. Execute per-index scene search with metadata filters.
4. Join results via plan join logic.
5. If empty -> none-analyzer creates plan deltas -> retry.
6. If non-empty -> validator filters candidates.
7. If all fail -> interpreter applies feedback deltas -> retry.
8. If accepted -> rerank -> preview.
9. Pause, persist state, and continue on follow-up.

## 13) Data Contracts

### 13.1 Plan Contract

- `subqueries[]`: `subquery_id`, `q`, `index[]`, `dialogue`
- `join_plan`: boolean join strategy
- `metadata_filters`: facet values
- `fallback_order`: facet relaxation order

### 13.2 Result Contract

- `clips[]` with `video_id`, `start`, `end`, score/rank fields, and explainability fields.
- `waiting_for`: user input or clarification when paused.

### 13.3 Feedback Contract

- validation verdict summary
- suggested delta operations
- operation reason codes

## 14) Reliability and Error Handling

- Structured error categories:
  - `auth_error`
  - `provider_error`
  - `timeout_error`
  - `missing_index_error`
  - `invalid_plan_error`
  - `validation_error`
- Retries with bounded backoff for model calls and remote operations.
- Idempotent indexing stages where possible.

## 15) Observability

- Node/stage-level logs for indexing and retrieval.
- Trace IDs across session/indexing lifecycle.
- Token and latency tracking per model stage.
- Optional telemetry adapter integration (e.g., Langfuse/OpenTelemetry).

## 16) Security

- Secrets via env/secret manager adapters only.
- No plaintext key persistence in stored session payloads.
- Redact keys and sensitive headers in logs/errors.

## 17) Performance Targets (Initial)

- Time-to-first-successful-indexed-query: < 15 minutes for standard sample videos.
- Retrieval turn latency target: p50 < 6s, p95 < 15s (excluding very large collections).
- Indexing success rate target: >= 95% on validated test corpus.

## 18) Testing Strategy

- Unit tests:
  - plan delta normalization/apply
  - routing and pause/resume behavior
  - provider adapter behavior
- Integration tests:
  - end-to-end indexing over sample media
  - retrieval loop with follow-up turns
  - model override behavior per stage
- Regression suite:
  - fixed benchmark queries and expected relevance bands

## 19) Milestones

### M1: Core Package Foundation

- Unified package structure for indexing + retrieval
- Config + provider interfaces
- session stores

### M2: Indexing v0.1

- Full pipeline from input to VideoDB index creation
- required index names + metadata facets coverage
- flexible model mapping by stage

### M3: Retrieval v0.1

- stateful loop, follow-up support, explainability output
- bounded retries and recovery behavior

### M4: Hardening

- observability, reliability improvements, docs and examples
- production validation with benchmark datasets

## 20) v0.1 Acceptance Criteria

- A developer can index a video and run multi-turn retrieval through Python API only.
- Default configuration works with Vercel AI SDK Python routing.
- OpenRouter can be enabled through config without code changes.
- Per-stage model configuration is applied correctly.
- Required indexes and retrieval facets are present and queryable.
- Indexing progress events are available through callback API.
- Session follow-up queries reuse state and improve/refine results.
- Explainability payload is returned for each ranked clip.

## 21) Final v0.1 Locks For Handoff

- Transcript path is configurable, with `gemini` as baseline default.
- LLM route support in v0.1 includes:
  - required: `vercel_ai_sdk_python`
  - optional: `openrouter`
- Provider and detector adapters are extension points; additional third-party platforms are post-v0.1.
