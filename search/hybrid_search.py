import asyncio
from typing import Any, Dict, List

from search.keyword_search import KeywordSearchService
from search.vector_search import VectorSearchService

# RRF 상수 — 순위 편향 방지용 (논문 권장값 60)
RRF_K = 60

# Exact Match Boost 보정값
BOOST_PHRASE = 0.5   # 구문 그대로 포함 시 +50%
BOOST_TOKEN  = 0.3   # 토큰 2개 이상 겹침 시 +30%


class HybridSearchService:
    """벡터 + Elasticsearch BM25 검색 결과를 RRF 알고리즘으로 융합하는 서비스"""

    def __init__(
        self,
        vector_search: VectorSearchService,
        keyword_search: KeywordSearchService,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> None:
        self._vector_search = vector_search
        self._keyword_search = keyword_search
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight

    async def search(
        self,
        query: str,
        brand: str,
        model: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """하이브리드 검색 — 벡터/키워드 병렬 실행 후 RRF 융합"""
        # 벡터 검색과 키워드 검색 병렬 실행
        vector_results, keyword_results = await asyncio.gather(
            self._vector_search.search(query, brand, model, top_k),
            self._keyword_search.search(query, brand, model, top_k),
        )

        # RRF 융합
        fused = self._reciprocal_rank_fusion(vector_results, keyword_results)

        # Exact Match Boost 보정
        boosted = self._apply_exact_match_boost(fused, query)

        # 최종 점수 내림차순 정렬 후 top_k 반환
        boosted.sort(key=lambda x: x["final_score"], reverse=True)
        return boosted[:top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """RRF 융합 — 절대 점수 대신 순위 기반으로 스케일 차이 극복

        score = vector_weight * Σ(1 / (k + rank)) + keyword_weight * Σ(1 / (k + rank))
        """
        scores: Dict[str, Dict[str, Any]] = {}

        # 벡터 검색 결과 순위 반영
        for rank, result in enumerate(vector_results):
            doc_id = result["id"]
            rrf_score = self._vector_weight * (1 / (RRF_K + rank + 1))
            if doc_id not in scores:
                scores[doc_id] = {**result, "final_score": 0.0}
            scores[doc_id]["final_score"] += rrf_score

        # 키워드 검색 결과 순위 반영
        for rank, result in enumerate(keyword_results):
            doc_id = result["id"]
            rrf_score = self._keyword_weight * (1 / (RRF_K + rank + 1))
            if doc_id not in scores:
                scores[doc_id] = {**result, "final_score": 0.0}
            scores[doc_id]["final_score"] += rrf_score

        return list(scores.values())

    def _apply_exact_match_boost(
        self,
        results: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """Exact Match Boost — 쿼리와 문서 텍스트 간 정확 매칭 시 점수 보정"""
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        for result in results:
            text_lower = result["text"].lower()
            text_tokens = set(text_lower.split())

            if query_lower in text_lower:
                # 구문 그대로 포함 시 +50%
                result["final_score"] *= (1 + BOOST_PHRASE)
            elif len(query_tokens & text_tokens) >= 2:
                # 토큰 2개 이상 겹침 시 +30%
                result["final_score"] *= (1 + BOOST_TOKEN)

        return results
