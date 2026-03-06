from __future__ import annotations

from typing import Dict, List, Set

from deepsearch.retrieval.helpers.schema import Shot, JoinedShot, ShotKey, JoinPlan


class Joiner:
    @staticmethod
    def _key(s: Shot) -> ShotKey:
        return (s.video_id, s.start, s.end)

    def _build_maps(self, index_sets: Dict[str, Dict]) -> Dict[str, Dict[ShotKey, Shot]]:
        per_map: Dict[str, Dict[ShotKey, Shot]] = {}
        for qid, payload in index_sets.items():
            shots = payload.get("shots", []) or []
            per_map[qid] = {self._key(s): s for s in shots}
        return per_map

    @staticmethod
    def _intersect(sets: List[Set[ShotKey]]) -> Set[ShotKey]:
        if not sets:
            return set()
        acc = sets[0].copy()
        for s in sets[1:]:
            acc &= s
        return acc

    @staticmethod
    def _union(sets: List[Set[ShotKey]]) -> Set[ShotKey]:
        acc: Set[ShotKey] = set()
        for s in sets:
            acc |= s
        return acc

    def _keys_for_qids(self, per_map: Dict[str, Dict[ShotKey, Shot]], qids: List[str]) -> List[Set[ShotKey]]:
        return [set(per_map.get(qid, {}).keys()) for qid in qids]

    def run(self, index_sets: Dict[str, Dict], join_plan: JoinPlan) -> Dict:
        per_map = self._build_maps(index_sets)
        all_qids = list(index_sets.keys())
        per_totals = {qid: len(index_sets.get(qid, {}).get("shots", []) or []) for qid in all_qids}

        op = join_plan.op or "OR"

        if join_plan.subqueries:
            qids = join_plan.subqueries
            sets = self._keys_for_qids(per_map, qids)
            final_keys = self._intersect(sets) if op == "AND" else self._union(sets)
        else:
            clauses = join_plan.clauses or []
            if not clauses:
                qids = all_qids
                sets = self._keys_for_qids(per_map, qids)
                final_keys = self._intersect(sets)
            else:
                clause_sets: List[Set[ShotKey]] = []
                for clause in clauses:
                    ksets = self._keys_for_qids(per_map, clause)
                    clause_keys = self._intersect(ksets) if op == "OR" else self._union(ksets)
                    clause_sets.append(clause_keys)
                final_keys = self._union(clause_sets) if op == "OR" else self._intersect(clause_sets)

        joined: List[JoinedShot] = []
        for key in sorted(final_keys):
            contributions = [(qid, per_map[qid][key]) for qid in all_qids if key in per_map.get(qid, {})]
            if not contributions:
                continue
            primary_qid, primary_shot = max(contributions, key=lambda p: (p[1].search_score or 0.0))
            texts = [(s.provenance.get("scene_index_name", ""), s.text) for _, s in contributions]
            js = JoinedShot(
                video_id=primary_shot.video_id,
                start=primary_shot.start,
                end=primary_shot.end,
                text=texts,
                primary={
                    "index": primary_shot.provenance.get("index"),
                    "subquery_id": primary_qid,
                    "variant_id": primary_shot.provenance.get("variant_id"),
                    "search_score": str(primary_shot.search_score) if primary_shot.search_score is not None else None,
                },
                support_subqueries=[qid for qid, _ in contributions if qid != primary_qid],
                metadata=primary_shot.metadata,
            )
            joined.append(js)

        joined.sort(key=lambda js: float(js.primary.get("search_score") or 0), reverse=True)

        return {
            "joined_shots": joined,
            "join_size": len(joined),
            "reason_code": {"code": "JOIN_RESULT" if joined else "JOIN_ZERO", "args": {"op": op, "join_size": len(joined), **per_totals}},
        }
