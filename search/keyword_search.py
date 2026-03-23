import os
from typing import Any, Dict, List

from elasticsearch import AsyncElasticsearch, NotFoundError
from elasticsearch.helpers import async_bulk

from store.vector_store import VectorStore

# Elasticsearch 연결 설정
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = "manuals"

# 검색 결과 최소 점수 임계값
# BM25 점수는 0~1 정규화가 아니므로 완전히 무관한 결과만 제거하는 낮은 값 사용
MIN_SCORE = 0.1

# ES 인덱스 매핑 — Nori 한국어 형태소 분석기
INDEX_SETTINGS = {
    "settings": {
        "analysis": {
            "analyzer": {
                "nori_analyzer": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "filter": ["nori_part_of_speech"],
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "text":  {"type": "text", "analyzer": "nori_analyzer"},
            "brand": {"type": "keyword"},
            "model": {"type": "keyword"},
        }
    },
}


class KeywordSearchService:
    """Elasticsearch BM25 기반 키워드 검색 서비스"""

    def __init__(self, es_url: str = ES_URL) -> None:
        self._es = AsyncElasticsearch(es_url)

    async def _ensure_index(self) -> None:
        """인덱스가 없으면 생성"""
        try:
            await self._es.indices.get(index=ES_INDEX)
        except NotFoundError:
            await self._es.indices.create(index=ES_INDEX, **INDEX_SETTINGS)

    async def search(
        self,
        query: str,
        brand: str,
        model: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """BM25 키워드 검색 — brand/model 필터 + 최소 점수 필터링"""
        await self._ensure_index()

        response = await self._es.search(
            index=ES_INDEX,
            size=top_k,
            min_score=MIN_SCORE,
            query={
                "bool": {
                    # brand/model 필터 — 해당 제품 청크만 검색
                    "filter": [
                        {"term": {"brand": brand}},
                        {"term": {"model": model}},
                    ],
                    # BM25 키워드 검색
                    "must": {
                        "match": {"text": query}
                    },
                }
            },
        )

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "text": hit["_source"]["text"],
                "metadata": {
                    "brand": hit["_source"]["brand"],
                    "model": hit["_source"]["model"],
                },
                "score": hit["_score"],
            })
        return results

    async def migrate_from_vector_store(self, vector_store: VectorStore) -> None:
        """ChromaDB 전체 문서를 ES에 일괄 인덱싱 — ES 초기 구축 시 사용"""
        await self._ensure_index()

        documents = vector_store.get_all_documents()
        actions = [
            {
                "_index": ES_INDEX,
                "_id": doc["id"],
                "_source": {
                    "text":  doc["text"],
                    "brand": doc["metadata"]["brand"],
                    "model": doc["metadata"]["model"],
                },
            }
            for doc in documents
        ]
        await async_bulk(self._es, actions)
