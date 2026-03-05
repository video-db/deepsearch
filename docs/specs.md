# DeepSearch Specs

Version: 0.1  
Status: Brainstorm Draft

## 1) Scope

DeepSearch is a Python library with two integrated capabilities:

1. Indexing orchestration
2. Stateful retrieval orchestration

DeepSearch uses VideoDB for extraction, storage, and search primitives.

## 2) Decisions Locked for v0.1

- Retrieval orchestration uses LangGraph.
- Indexing is blocking and local-process only.
- Indexing exposes stage progress through local event callbacks.
- Session persistence is pluggable.
- No name-disambiguation clarification branch in retrieval routing.
- LLM routing default is Vercel AI SDK Python.
- OpenRouter is optional.
- Model selection is configurable per stage.
- Detector integration is provider-style and extensible to third-party services.

## 3) Core Dependencies

- `langgraph`: retrieval graph execution and pause/resume orchestration
- `pydantic`: typed contracts for config/state/stage I/O
- `videodb` (videodb-python): media operations and index/search APIs
- Vercel AI SDK Python runtime (default LLM route)

## 4) Package Layout (Proposed)

```text
deepsearch/
  client/
    client.py
    sessions.py
  indexing/
    pipeline.py
    manifest.py
    contracts.py
    stages/
      extract.py
      transcript.py
      detect.py
      enrich.py
      summarize.py
      write_indexes.py
  retrieval/
    graph.py
    state.py
    contracts.py
    nodes/
      plan_init.py
      search_join.py
      validator.py
      none_analyzer.py
      interpreter.py
      reranker.py
    delta/
  providers/
    llm/
      base.py
      vercel_ai_sdk.py
      openrouter.py
      adapters/
    detector/
      base.py
      local_rtdetr.py
      local_yolo.py
      adapters/
  stores/
    session_store.py
    memory.py
    sqlite.py
  config/
    schema.py
    defaults.py
  telemetry/
    tracing.py
    logger.py
  errors/
    codes.py
```

## 5) Public Python API Contract

```python
from deepsearch import DeepSearchClient

client = DeepSearchClient(config=...)

# Indexing (blocking)
manifest = client.index_video(
    video_url="...",
    collection_id="...",
    on_event=lambda event: print(event.stage, event.status),
)

# Retrieval
session = client.start_session(collection_id="...", video_id=None)
result = session.search("find rainy running scenes")
result = session.followup("only close-up shots")

# Resume
session2 = client.resume_session(session_id=result.session_id)
```

Constructor contract:

- `DeepSearchClient(config, session_store=None, index_progress_store=None)`
- `config` may be: typed config object, dict, or path to config file.
- `session_store` must satisfy `SessionStore` protocol.
- `index_progress_store` is optional and must satisfy `IndexProgressStore` protocol.

Required methods:

- `DeepSearchClient.index_video(...) -> IndexManifest`
- `DeepSearchClient.start_session(collection_id, video_id=None, page_size=None) -> DeepSearchSession`
- `DeepSearchClient.resume_session(session_id) -> DeepSearchSession`
- `DeepSearchSession.search(query, *, page_size=None) -> RetrievalResult`
- `DeepSearchSession.followup(text=None, ui_event=None, page_size=None) -> RetrievalResult`

Indexing signatures:

- `index_video(..., video_url: str | None = None, media_id: str | None = None, on_event: Callable[[IndexEvent], None] | None = None) -> IndexManifest`

Input constraints:

- `index_video`: exactly one of `video_url` or `media_id` is required.
- `followup`: at least one of `text` or `ui_event` is required.

## 6) Typed Contracts

### 6.1 IndexRequest

```python
class IndexRequest(BaseModel):
    collection_id: str
    video_url: str | None = None
    media_id: str | None = None
    config_overrides: dict[str, Any] = {}
```

Constraints:

- exactly one of `video_url` or `media_id` must be set

```python
class IndexArtifact(BaseModel):
    index_name: str
    index_id: str
    scene_count: int

class StageTiming(BaseModel):
    stage: str
    duration_ms: int

class ReplacedIndexRef(BaseModel):
    index_name: str
    old_index_id: str
    new_index_id: str

class IndexStats(BaseModel):
    total_scenes: int
    stage_timings: list[StageTiming] = []
    token_usage: dict[str, int] = {}
    replaced_indexes: list[ReplacedIndexRef] = []
```

### 6.2 IndexManifest

```python
class IndexManifest(BaseModel):
    manifest_id: str
    collection_id: str
    video_id: str
    indexes: dict[str, IndexArtifact]
    stats: IndexStats
    stage_statuses: list[IndexStageStatus]
    warnings: list[str] = []
```

`manifest_id` is created at pipeline start and is stable across all stage events.

`indexes` must include all required index names:

- `location`
- `scene_description`
- `transcript`
- `topic`
- `object_description`
- `subplot_summary`
- `final_summary`

### 6.3 IndexEvent

```python
class IndexEvent(BaseModel):
    stage: Literal[
        "extract",
        "transcript",
        "detect",
        "enrich",
        "summarize",
        "write_indexes",
        "manifest",
    ]
    status: Literal["started", "completed", "failed"]
    message: str | None = None
    progress: float | None = None
    ts: datetime

class IndexStageStatus(BaseModel):
    stage: str
    status: Literal["pending", "running", "completed", "failed"]
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: float | None = None
    message: str | None = None
```

`IndexEvent.progress` is normalized to `[0, 100]`.

`IndexStageStatus` stores the final per-stage snapshot in `IndexManifest.stage_statuses`.

```python
class UiEvent(BaseModel):
    type: str
    payload: dict[str, Any] = {}

class ClarificationOption(BaseModel):
    id: str
    label: str

class ClarificationQuestion(BaseModel):
    question_id: str
    text: str
    mode: Literal["text", "mcq"]
    options: list[ClarificationOption] = []
```

Reserved `UiEvent.type` values for v0.1:

- `show_more`
- `clarification_answer`
- `by_example`

### 6.4 RetrievalResult

```python
class RetrievalResult(BaseModel):
    session_id: str
    clips: list[ClipResult]
    waiting_for: Literal["user_input", "clarification", "none"]
    clarification: ClarificationQuestion | None = None
    page: PageInfo
    debug: dict[str, Any] | None = None
```

`ClipResult` required fields:

- `video_id`, `start`, `end`
- `stream_url` (or equivalent playable reference)
- `rank`, `score`
- `explain.primary_subquery`
- `explain.primary_index`
- `explain.support_subqueries[]`

Clip ordering contract:

- `clips[]` are returned in ranked order.
- `rank` is 1-based within the current page response.

`PageInfo` contract:

```python
class PageInfo(BaseModel):
    page_size: int
    cursor: int
    next_cursor: int | None
    has_more: bool
```

### 6.5 Retrieval Plan

```python
class Subquery(BaseModel):
    subquery_id: str
    q: str
    index: list[str]
    dialogue: bool = False

class JoinPlan(BaseModel):
    op: Literal["AND", "OR"]
    subqueries: list[str] | None = None
    clauses: list[list[str]] | None = None

class Plan(BaseModel):
    subqueries: list[Subquery]
    join_plan: JoinPlan
    metadata_filters: dict[str, list[str]]
    fallback_order: list[str]
```

Plan constraints:

- each `subquery_id` must be unique.
- `join_plan` must reference existing `subquery_id` values.
- `join_plan` must use exactly one of `subqueries` or `clauses`.
- `metadata_filters` keys are limited to supported retrieval facets.
- `fallback_order` values must be unique and within supported retrieval facets.

v0.1 supported metadata facets:

- `shot_type`
- `emotion`
- `objects`

### 6.6 Delta Operation Contract

```python
class DeltaOp(BaseModel):
    op: Literal[
        "set_filter",
        "merge_filter",
        "drop_filter",
        "drop_values",
        "set_subquery",
        "merge_indexes",
        "set_join_plan",
        "by_example",
    ]
    payload: dict[str, Any]
```

Delta constraints:

- max ops per cycle is controlled by `max_ops_per_batch`.
- facet operations can only target supported metadata facets.
- all operations must be normalized and validated before apply.

## 7) Indexing Pipeline Stage Contracts

All indexing stages use typed input/output contracts.

All stage outputs must be JSON-serializable and SDK-agnostic (no raw VideoDB objects in persisted artifacts).

Stage execution order is fixed in v0.1:

1. `extract`
2. `transcript`
3. `detect`
4. `enrich`
5. `summarize`
6. `write_indexes`
7. `manifest`

Event contract per stage:

- emit `IndexEvent(status="started")` when stage begins.
- emit exactly one terminal event: `completed` or `failed`.
- update `IndexManifest.stage_statuses` from emitted events.
- if a stage is config-disabled (for example `summarize`), emit `started` and `completed` with `message="skipped"`.

Failure behavior:

- pipeline is fail-fast on first stage failure.
- `index_video` raises `DeepSearchError(code="DS_PIPELINE_STAGE_ERROR")`.
- partial writes during `write_indexes` may exist; rerun with overwrite policy must reconcile to a canonical final state.

Canonical artifact models:

```python
class SceneRef(BaseModel):
    scene_id: str
    video_id: str
    start: float
    end: float
    frame_urls: list[str]

class FrameRef(BaseModel):
    frame_time: float
    frame_url: str

class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str

class Detection(BaseModel):
    label: str
    score: float
    box: list[float] | None = None

class VisionFrameMeta(BaseModel):
    frame_time: float
    detections: list[Detection]

class CompiledScene(BaseModel):
    video_id: str
    start: float
    end: float
    location: str | None = None
    scene_description: str | None = None
    transcript: str | None = None
    topic: str | None = None
    object_description: str | None = None
    shot_type: str | None = None
    emotion: str | None = None
    objects: list[str] = []

class SubplotSegment(BaseModel):
    start: float
    end: float
    summary: str
```

### 7.1 Stage 1: Extract

- input: `video_id`, extraction config
- output:
  - `shot_scene_collection_id`
  - `time_scene_collection_id`
  - `shot_scenes[]` as `SceneRef`
  - `time_scenes[]` as `SceneRef`
  - extraction stats

### 7.2 Stage 2: Transcript

- input: `video_id`, transcript config
- output:
  - transcript segments as `TranscriptSegment[]`
  - transcript stats

### 7.3 Stage 3: Detect

- input: frame references as `FrameRef[]` from `time_scene_collection_id`
- output:
  - `vision_metadata[]` as `VisionFrameMeta[]`
  - detection stats

### 7.4 Stage 4: Enrich

- input:
  - shot scenes
  - transcript segments
  - vision metadata
- output:
  - `compiled_scenes[]` as `CompiledScene[]`
  - LLM usage stats

### 7.5 Stage 5: Summarize

- input: `compiled_scenes[]`
- output:
  - `subplots[]` as `SubplotSegment[]`
  - `final_summary`

### 7.6 Stage 6: Write Indexes

- input:
  - `compiled_scenes[]`
  - optional subplot/final summary data
- output:
  - created index ids keyed by index name
  - index write stats

Default overwrite policy (v0.1):

- For same `video_id` and same index name, replace existing index with newly generated index.
- Manifest must record replaced index ids in `stats.replaced_indexes[]`.

### 7.7 Stage 7: Manifest

- input: all stage outputs
- output: `IndexManifest`

Manifest validation rules:

- all required index names must be present in `manifest.indexes`.
- missing required index -> raise `DeepSearchError(code="DS_MISSING_INDEX_ERROR")`.

## 8) Index Writing Strategy (Why and How)

### 8.1 Field-to-Index Mapping (v0.1)

- `location` index <- `CompiledScene.location`
- `scene_description` index <- `CompiledScene.scene_description`
- `transcript` index <- `CompiledScene.transcript`
- `topic` index <- `CompiledScene.topic`
- `object_description` index <- `CompiledScene.object_description`
- `subplot_summary` index <- `SubplotSegment.summary`
- `final_summary` index <- pipeline `final_summary`

### 8.2 Writer Behavior

DeepSearch uses one deterministic writer path in v0.1:

1. Build one canonical `compiled_scenes[]` representation.
2. Derive per-index scene records from that canonical data.
3. Call VideoDB scene index write APIs for each index name.
4. Replace existing same-name indexes for the same video.

Why this is required:

- Keeps scene boundaries consistent across indexes.
- Keeps metadata (`shot_type`, `emotion`, `objects`) aligned across indexes.
- Avoids duplicated extraction/model work.

Example:

- Bad path (avoid): call separate model workflows for `location` and `scene_description`; scene boundaries drift and joins get noisy.
- Chosen path: one canonical scene at `start=42.0,end=46.0` yields both `location` and `scene_description` index entries with shared metadata, so retrieval joins are stable.

## 9) Retrieval Graph Spec (LangGraph)

### 9.1 Nodes

- `plan_init`
- `search_join`
- `validator`
- `none_analyzer`
- `interpreter`
- `reranker`
- `preview_pause`
- `clarify_pause`

### 9.2 Entry Routing

- Fresh session turn (`paused_for is None`) -> `plan_init`
- Resume turn (`paused_for is not None`) -> `interpreter`

No name-disambiguation clarification branch exists in v0.1.

### 9.3 Core Transitions

- `search_join` empty -> `none_analyzer`
- `search_join` non-empty -> `validator`
- `validator` accepted -> `reranker`
- `validator` all-fail -> `interpreter`
- `interpreter` delta batch -> `search_join`
- `interpreter` clarification -> `clarify_pause`
- `reranker` -> `preview_pause`

### 9.4 Safety Controls

- `recursion_limit` hard cap (default 12)
- validator retry/reset caps
- none-analyzer retry/reset caps
- `max_ops_per_batch`

### 9.5 `show_more` Semantics (Locked)

- `show_more` does not trigger a new retrieval pass.
- It paginates existing ranked results using `page.cursor` and `page_size`.
- If exhausted, return `clips=[]`, `has_more=false`, `next_cursor=None`, `waiting_for="user_input"`.
- Exhaustion response should include a stable hint value: `debug.status_hint="no_more_results"`.

Pagination rules:

- first successful search/follow-up retrieval response starts with `cursor=0`.
- `next_cursor` advances by number of emitted clips.
- `page_size` precedence: method argument > session default > config default.

## 10) Provider Contracts

### 10.1 LLM Provider Contract

```python
class LLMProvider(Protocol):
    def generate_json(self, *, task: str, prompt: str, model: str, options: dict[str, Any]) -> LLMJsonResponse: ...
    def generate_text(self, *, task: str, prompt: str, model: str, options: dict[str, Any]) -> LLMTextResponse: ...
```

Response contract:

```python
class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None

class LLMJsonResponse(BaseModel):
    data: dict[str, Any]
    usage: Usage
    raw_model_id: str
```

Requirements:

- pass provider-specific options through `options`
- normalize usage accounting
- map provider errors to DeepSearch error taxonomy

Registration contract:

- `register_llm_provider(name: str, factory: Callable[[dict[str, Any]], LLMProvider])`
- config selects provider by name via `llm.route` and `llm.provider_mode`.

Default provider route:

- `vercel_ai_sdk_python` (required)
- `openrouter` (optional)

### 10.2 Detector Provider Contract

```python
class DetectorProvider(Protocol):
    def detect_batch(self, frames: list[FrameRef], config: dict[str, Any]) -> list[FrameDetections]: ...
```

`FrameDetections` contract:

```python
class Detection(BaseModel):
    label: str
    score: float
    box: list[float] | None = None

class FrameDetections(BaseModel):
    frame_time: float
    detections: list[Detection]
    provider: str
```

Requirements:

- normalize response schema independent of backend
- support both built-in local detectors and third-party detector adapters

Registration contract:

- `register_detector_provider(name: str, factory: Callable[[dict[str, Any]], DetectorProvider])`
- config selects provider by name via `indexing.object_detection.provider`.

## 11) Config Schema (v0.1)

### 11.1 LLM

- `llm.route`: default `vercel_ai_sdk_python`
- `llm.provider_mode`: default `direct`
- `llm.openrouter.enabled`: default `false`
- `llm.models.indexing.*` and `llm.models.retrieval.*` per-stage mapping

### 11.2 Indexing Defaults

- shot extraction: threshold `30`, frame_count `10`
- time extraction: time `1`, frame_count `10`
- transcript path: spoken-word indexing, baseline `gemini`
- detector baseline: `rtdetr_v2`, threshold `0.85`, batch_size `64`
- indexing model baseline: `o3`
- `overwrite_existing_indexes=true`

### 11.3 Retrieval Defaults

- retrieval model baseline: `gpt-4o-2024-11-20`
- `k_variants_per_index=2`
- `topk_per_variant=30`
- `validator_max=40`
- `validator_batch_size=8`
- `max_ops_per_batch=4`
- `page_size=10`
- `recursion_limit=12`

### 11.4 Config Resolution Precedence

Highest to lowest precedence:

1. per-call `config_overrides` in `index_video(...)`
2. constructor config values
3. environment variables
4. package defaults

Resolved config should be frozen at run start and reused across all stages in that run.

### 11.5 Typed Config API (recommended)

```python
class LLMIndexingModelsConfig(BaseModel):
    scene_enrichment: str = "o3"
    subplot_summary: str = "o3"
    final_summary: str = "o3"

class LLMRetrievalModelsConfig(BaseModel):
    planner: str = "gpt-4o-2024-11-20"
    paraphrase: str = "gpt-4o-2024-11-20"
    validator: str = "gpt-4o-2024-11-20"
    none_analyzer: str = "gpt-4o-2024-11-20"
    interpreter: str = "gpt-4o-2024-11-20"
    reranker: str = "gpt-4o-2024-11-20"

class LLMModelsConfig(BaseModel):
    indexing: LLMIndexingModelsConfig = LLMIndexingModelsConfig()
    retrieval: LLMRetrievalModelsConfig = LLMRetrievalModelsConfig()

class LLMConfig(BaseModel):
    route: str = "vercel_ai_sdk_python"
    provider_mode: str = "direct"
    models: LLMModelsConfig = LLMModelsConfig()

class RetrievalConfig(BaseModel):
    page_size: int = 10
    k_variants_per_index: int = 2
    topk_per_variant: int = 30
    validator_max: int = 40
    validator_batch_size: int = 8
    max_ops_per_batch: int = 4
    recursion_limit: int = 12

class DeepSearchConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    indexing: dict[str, Any] = {}
    retrieval: RetrievalConfig = RetrievalConfig()

    @classmethod
    def defaults(cls) -> "DeepSearchConfig": ...

    @classmethod
    def from_file(cls, path: str) -> "DeepSearchConfig": ...

    @classmethod
    def from_env(cls, prefix: str = "DEEPSEARCH_") -> "DeepSearchConfig": ...

    def with_overrides(self, overrides: dict[str, Any]) -> "DeepSearchConfig": ...
```

Client constructor must accept:

- typed config object (`DeepSearchConfig`)
- plain dict
- path to config file

## 12) Session Persistence Contract

Persisted state payload (minimum):

```python
class PersistedSessionState(BaseModel):
    session_id: str
    collection_id: str
    video_id: str | None = None
    main_query: str
    plan: Plan | None = None
    history: list[dict[str, Any]] = []
    paused_for: Literal["preview_pause", "clarify_pause", None] = None
    ranked_cache: list[ClipResult] = []
    page_cursor: int = 0
```

State payload must remain JSON-serializable.

```python
class SessionStore(Protocol):
    def save_state(self, session_id: str, state: dict[str, Any]) -> None: ...
    def load_state(self, session_id: str) -> dict[str, Any] | None: ...
    def delete_state(self, session_id: str) -> None: ...
```

Implementations:

- required: in-memory
- optional: sqlite

SQLite is an adapter option; not a required dependency for v0.1.

### 12.2 Index Progress Store Contract (optional)

```python
class IndexProgressStore(Protocol):
    def append_event(self, manifest_id: str, event: IndexEvent) -> None: ...
    def list_events(self, manifest_id: str) -> list[IndexEvent]: ...
    def latest_stage_status(self, manifest_id: str) -> list[IndexStageStatus]: ...
```

If enabled, sqlite can implement `IndexProgressStore` for local status inspection of each indexing stage.

## 13) Error Taxonomy

Required error codes:

- `DS_AUTH_ERROR`
- `DS_PROVIDER_ERROR`
- `DS_TIMEOUT_ERROR`
- `DS_MISSING_INDEX_ERROR`
- `DS_INVALID_PLAN_ERROR`
- `DS_VALIDATION_ERROR`
- `DS_PIPELINE_STAGE_ERROR`

Error payload contract:

- `code`
- `message`
- `stage_or_node`
- `retryable`
- `details` (optional)

Public API behavior:

- invalid user input contracts (e.g., missing `video_url/media_id`, empty follow-up) raise `ValueError`.
- runtime failures raise `DeepSearchError` with one of the required error codes.
- retrieval errors must preserve `session_id` and last persisted state.

```python
class DeepSearchError(Exception):
    code: str
    message: str
    stage_or_node: str
    retryable: bool
    details: dict[str, Any] | None = None
```

## 14) Observability Contract

Minimum telemetry fields:

- `trace_id`
- `session_id`
- `stage_or_node`
- latency ms
- model id
- token usage

Telemetry is optional and must not block execution.

## 15) Test Plan

### 15.1 Unit

- config precedence
- stage contract validation
- delta normalization and apply
- provider adapter contract behavior

### 15.2 Integration

- indexing e2e on fixture videos
- retrieval multi-turn with session resume
- per-stage model override checks
- pagination contract checks for `show_more`

### 15.3 Regression

- benchmark query suite with acceptance bands for relevance and stability

## 16) Acceptance Criteria (v0.1)

- Indexing and retrieval work through Python API.
- Indexing is blocking and returns `IndexManifest` directly.
- Indexing emits typed local progress events (`IndexEvent`) when callback is provided.
- Retrieval orchestration uses LangGraph with the defined node graph.
- Required index names are produced and queryable.
- Required retrieval facets (`shot_type`, `emotion`, `objects`) are supported.
- Session persistence works with in-memory store and sqlite adapter.
- `show_more` uses pagination over existing ranked results.
- Result clips include explainability fields.
