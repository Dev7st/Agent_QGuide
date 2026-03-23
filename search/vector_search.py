from typing import Any, Dict, List

from store.embedding import EmbeddingService
from store.vector_store import VectorStore

# 유사도 임계값 — 이 값 미만의 결과는 관련 없는 문서로 판단하여 제외
SIMILARITY_THRESHOLD = 0.3


class VectorSearchService:
    """ChromaDB 벡터 검색 서비스 — 쿼리 임베딩 후 유사 청크 검색"""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    async def search(
        self,
        query: str,
        brand: str,
        model: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """벡터 유사도 검색

        top_k * 2 오버패치 후 유사도 임계값 필터링 적용.
        임계값 필터링으로 결과가 줄어도 최소한 top_k개를 확보하기 위한 전략.
        """
        # 쿼리 임베딩 생성
        query_embedding = await self._embedding_service.embed_query(query)

        # top_k * 2 오버패치 — 임계값 필터링 후 결과 부족 방지
        candidates = self._vector_store.search_similar_chunks(
            query_embedding=query_embedding,
            brand=brand,
            model=model,
            top_k=top_k * 2,
        )

        # 유사도 임계값 필터링 후 상위 top_k 반환
        results = [
            chunk for chunk in candidates
            if chunk["similarity"] >= SIMILARITY_THRESHOLD
        ]
        return results[:top_k]
