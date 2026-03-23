import os
from typing import Any, Dict, List

import chromadb

# ChromaDB 저장 경로
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "database/chroma_db")
COLLECTION_NAME = "manuals"


class VectorStore:
    """ChromaDB 연결 및 벡터 저장/검색 서비스"""

    def __init__(self, persist_dir: str = CHROMA_DB_PATH) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        # L2 거리 메트릭 지정 — normalize_embeddings=True와 조합 시 코사인 유사도와 동치
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "l2"},
        )

    def add_manual_embeddings(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """매뉴얼 청크 저장 — 기존 문서 삭제 후 재삽입 (잔여 청크 오염 방지)"""
        if not documents:
            return

        brand = documents[0]["metadata"]["brand"]
        model = documents[0]["metadata"]["model"]
        self._remove_existing_manual(brand, model)

        self._collection.add(
            ids=[doc["id"] for doc in documents],
            documents=[doc["text"] for doc in documents],
            embeddings=embeddings,
            metadatas=[doc["metadata"] for doc in documents],
        )

    def _remove_existing_manual(self, brand: str, model: str) -> None:
        """동일 brand/model의 기존 청크 전체 삭제"""
        results = self._collection.get(
            where={"$and": [{"brand": brand}, {"model": model}]}
        )
        if results["ids"]:
            self._collection.delete(ids=results["ids"])

    def search_similar_chunks(
        self,
        query_embedding: List[float],
        brand: str,
        model: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """벡터 유사도 검색 — L2 거리를 코사인 유사도(0~1)로 변환하여 반환"""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"$and": [{"brand": brand}, {"model": model}]},
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "metadata": metadata,
                "similarity": 1 / (1 + distance),  # L2 → 코사인 유사도 변환
            })
        return chunks

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """전체 문서 조회 — ES 초기 인덱싱(마이그레이션) 시드 데이터 제공"""
        results = self._collection.get(include=["documents", "metadatas"])
        documents = []
        for doc_id, text, metadata in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
        ):
            documents.append({
                "id": doc_id,
                "text": text,
                "metadata": metadata,
            })
        return documents
