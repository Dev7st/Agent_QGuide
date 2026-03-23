import asyncio
import os
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import List

from sentence_transformers import SentenceTransformer

# CVE-2025-32434: torch.load pickle 역직렬화 취약점 방어
_original_torch_load = torch.load


def _safe_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", True)
    return _original_torch_load(*args, **kwargs)


torch.load = _safe_torch_load

# 임베딩 모델 우선순위 (한국어 특화 → 다국어 → 범용)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")
FALLBACK_MODELS = [
    "paraphrase-multilingual-MiniLM-L12-v2",
    "all-MiniLM-L6-v2",
]


class EmbeddingService:
    """임베딩 모델 로드 및 텍스트 → 벡터 변환 서비스"""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.model = self._load_model()

    def _load_model(self) -> SentenceTransformer:
        """모델 로드 — 실패 시 Fallback 순서대로 재시도"""
        candidates = [EMBEDDING_MODEL] + FALLBACK_MODELS
        for model_name in candidates:
            try:
                model = SentenceTransformer(model_name)
                print(f"[EmbeddingService] 모델 로드 성공: {model_name}")
                return model
            except Exception as e:
                print(f"[EmbeddingService] 모델 로드 실패 ({model_name}): {e}")
        raise RuntimeError("사용 가능한 임베딩 모델이 없습니다.")

    def _encode_sync(self, texts: List[str]) -> List[List[float]]:
        """동기 인코딩 (ThreadPoolExecutor 내부에서 실행)"""
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    async def embed_query(self, query: str) -> List[float]:
        """단일 쿼리 임베딩 생성"""
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(self._executor, self._encode_sync, [query])
        return vectors[0]

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """다중 텍스트 임베딩 생성"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._encode_sync, texts)
