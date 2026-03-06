from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import videodb

from langgraph.types import Command

from deepsearch.retrieval.helpers.schema import Plan, Shot
from deepsearch.retrieval.helpers.join import Joiner
from deepsearch.retrieval.state import GraphState

logger = logging.getLogger(__name__)


class IndexClient:
    def __init__(self, video_id: str = None):
        self.video_id = video_id
        self.num_db_calls = 0

    def search(
        self,
        coll,
        index_list: list[str],
        q: str,
        metadata_filters: Dict[str, List[str]],
        topk: int,
        sid: Optional[str],
        score_threshold: float,
        dynamic_score_percentage: int,
    ) -> List[Shot]:
        mf = dict(metadata_filters)
        mf["scene_index_name"] = index_list
        mf = {k: v for k, v in mf.items() if v}

        try:
            if self.video_id:
                video = coll.get_video(self.video_id)
                results = video.search(
                    query=q,
                    index_type=videodb.IndexType.scene,
                    score_threshold=score_threshold,
                    dynamic_score_percentage=dynamic_score_percentage,
                    result_threshold=topk,
                    sort_docs_on="score",
                    search_type="semantic",
                    filter=[mf],
                    rerank=False,
                )
            else:
                results = coll.search(
                    query=q,
                    index_type=videodb.IndexType.scene,
                    score_threshold=score_threshold,
                    dynamic_score_percentage=dynamic_score_percentage,
                    result_threshold=topk,
                    sort_docs_on="score",
                    search_type="semantic",
                    filter=[mf],
                    rerank=False,
                )
            self.num_db_calls += 1

            dropped_without_video_id = 0
            shots: List[Shot] = []
            sample_dropped_hit: Optional[Any] = None
            for hit in results.shots:
                video_id = getattr(hit, "video_id", None)
                if not video_id:
                    # print(hit)
                    dropped_without_video_id += 1
                    sample_dropped_hit = sample_dropped_hit or hit
                    continue
                title = getattr(hit, "video_title", None) or ""
                shots.append(
                    Shot(
                        video_id=str(video_id),
                        video_title=str(title),
                        start=float(getattr(hit, "start", 0.0) or 0.0),
                        end=float(getattr(hit, "end", 0.0) or 0.0),
                        text=str(getattr(hit, "text", "") or ""),
                        search_score=float(getattr(hit, "search_score", 0.0) or 0.0),
                        provenance={
                            "scene_index_name": str(
                                getattr(hit, "scene_index_name", "") or ""
                            )
                        },
                        metadata=getattr(hit, "metadata", {}) or {},
                    )
                )
            if dropped_without_video_id:
                logger.warning(
                    "Dropped %s search hits without resolvable video_id",
                    dropped_without_video_id,
                )
                if sample_dropped_hit is not None:
                    logger.debug(
                        "Sample dropped hit attrs=%s metadata=%s",
                        {
                            "video_id": getattr(sample_dropped_hit, "video_id", None),
                            "scene_index_name": getattr(
                                sample_dropped_hit, "scene_index_name", None
                            ),
                            "start": getattr(sample_dropped_hit, "start", None),
                            "end": getattr(sample_dropped_hit, "end", None),
                        },
                        getattr(sample_dropped_hit, "metadata", None),
                    )
            return shots
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []


class ParaphraserLLM:
    def __init__(self, llm, prompts):
        self.llm = llm
        self.prompts = prompts

    def run(self, query: str, k: int, main_query: str):
        prompt = self.prompts.build_paraphrase_prompt(query, k, main_query)
        data, inp, out, total = self.llm.generate(prompt)
        out_list = (data or {}).get("paraphrases") or []
        return (
            [p for p in out_list if isinstance(p, str) and p.strip()][:k],
            inp,
            out,
            total,
        )


class ParaphraseSearch:
    def __init__(
        self,
        coll,
        index_client: IndexClient,
        k_variants: int = 2,
        topk: int = 30,
        paraphraser: Optional[ParaphraserLLM] = None,
    ):
        self.client = index_client
        self.k_variants = max(1, k_variants)
        self.topk = topk
        self.paraphraser = paraphraser
        self.coll = coll

    def run(
        self,
        plan: Plan,
        sid: str = None,
        main_query: str = None,
        score_threshold: float = 0.0,
        dynamic_score_percentage: int = 100,
    ):
        total_inp = total_out = total_total = 0
        per_subq_fused: Dict[str, Dict] = {}

        for sq in plan.subqueries:
            variants = [sq.q]
            if self.paraphraser:
                try:
                    v, inp, out, total = self.paraphraser.run(
                        sq.q, self.k_variants, main_query or ""
                    )
                    if v:
                        variants = v
                    total_inp += inp
                    total_out += out
                    total_total += total
                except Exception:
                    pass

            index_list = list(sq.index) if isinstance(sq.index, list) else [sq.index]
            fused_map: Dict[tuple, Shot] = {}

            if sq.dialogue and sq.dialogue.lower() == "true":
                variants.append(sq.q)

            for i, pv in enumerate(variants, start=1):
                shots = (
                    self.client.search(
                        self.coll,
                        index_list,
                        pv,
                        plan.metadata_filters,
                        self.topk,
                        sid,
                        score_threshold,
                        dynamic_score_percentage,
                    )
                    or []
                )
                for s in shots:
                    sc = s.model_copy(deep=True)
                    scene_index_name = sc.provenance.get("scene_index_name")
                    sc.provenance.update(
                        {
                            "index": str(
                                scene_index_name
                                or (
                                    index_list[0]
                                    if len(index_list) == 1
                                    else index_list
                                )
                            ),
                            "subquery_id": sq.subquery_id,
                            "variant_id": f"V{i}",
                        }
                    )
                    key = (sc.video_id, sc.start, sc.end)
                    best = fused_map.get(key)
                    if best is None or (sc.search_score or 0) > (
                        best.search_score or 0
                    ):
                        fused_map[key] = sc

            fused_shots = sorted(
                fused_map.values(), key=lambda s: s.search_score or 0, reverse=True
            )
            per_subq_fused[sq.subquery_id] = {
                "index": index_list,
                "shots": fused_shots,
                "total": len(fused_shots),
            }

        return {"index_sets": per_subq_fused}, total_inp, total_out, total_total


def search_join_node(state: GraphState):
    logger.info("search_join_node: searching")
    coll = state.collection
    cfg = state.cfg
    if cfg.debug_mode:
        logger.debug("SearchJoin plan=%s", state.plan.model_dump())
        logger.debug(
            "SearchJoin config k_variants_per_index=%s topk_per_variant=%s score_threshold=%s dynamic_score_percentage=%s",
            cfg.k_variants_per_index,
            cfg.topk_per_variant,
            cfg.score_threshold,
            cfg.dynamic_score_percentage,
        )

    index_client = IndexClient(video_id=state.video_id)
    paraphraser = ParaphraserLLM(state.llm_for("paraphrase"), state.prompts)
    searcher = ParaphraseSearch(
        coll,
        index_client,
        k_variants=cfg.k_variants_per_index,
        topk=cfg.topk_per_variant,
        paraphraser=paraphraser,
    )

    search_out, _, _, _ = searcher.run(
        state.plan,
        sid=state.session_id,
        main_query=state.main_query,
        score_threshold=cfg.score_threshold,
        dynamic_score_percentage=cfg.dynamic_score_percentage,
    )
    join_out = Joiner().run(search_out["index_sets"], state.plan.join_plan)

    joined = join_out["joined_shots"]
    logger.info(f"search_join_node: {len(joined)} joined shots")

    goto = "none_analyzer" if len(joined) == 0 else "validator"
    return Command(update={"joined_shots": joined}, goto=goto)
