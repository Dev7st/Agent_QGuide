from typing import Any, Dict, List

from langchain_core.tools import tool

from crawling.crawler import ManualCrawler
from search.hybrid_search import HybridSearchService
from search.keyword_search import KeywordSearchService
from search.vector_search import VectorSearchService
from store.embedding import EmbeddingService
from store.vector_store import VectorStore

# 모듈 레벨 싱글턴 — 최초 호출 시 초기화
_hybrid_search: HybridSearchService | None = None
_crawler: ManualCrawler | None = None


def _get_hybrid_search() -> HybridSearchService:
    """HybridSearchService 싱글턴 반환 (최초 호출 시 초기화)"""
    global _hybrid_search
    if _hybrid_search is None:
        embedding       = EmbeddingService()
        vector_store    = VectorStore()
        vector_search   = VectorSearchService(embedding, vector_store)
        keyword_search  = KeywordSearchService()
        _hybrid_search  = HybridSearchService(vector_search, keyword_search)
    return _hybrid_search


def _get_crawler() -> ManualCrawler:
    """ManualCrawler 싱글턴 반환 (최초 호출 시 초기화)"""
    global _crawler
    if _crawler is None:
        _crawler = ManualCrawler()
    return _crawler


def _format_results(results: List[Dict[str, Any]]) -> str:
    """검색 결과 리스트를 번호 매긴 텍스트로 변환"""
    return "\n".join(f"[{i + 1}] {r['text']}" for i, r in enumerate(results))


@tool
async def manual_search(query: str, brand: str, model: str) -> str:
    """삼성 세탁기 매뉴얼에서 관련 정보를 검색한다.

    매뉴얼 정보가 필요할 때 가장 먼저 호출한다.
    검색 결과가 없으면 manual_crawl을 호출하여 매뉴얼을 수집한다.

    Args:
        query: 사용자 질문 (예: "E3 오류코드 해결법")
        brand: 세탁기 브랜드 (samsung)
        model: 세탁기 모델명 (예: WF-85A)

    Returns:
        검색된 매뉴얼 청크 텍스트 또는 "해당 제품의 매뉴얼이 등록되어 있지 않습니다."
    """
    results = await _get_hybrid_search().search(query, brand, model)

    if not results:
        return "해당 제품의 매뉴얼이 등록되어 있지 않습니다."

    return _format_results(results)


@tool
async def manual_crawl(brand: str, model: str) -> str:
    """삼성 세탁기 매뉴얼을 제조사 사이트에서 수집하여 DB에 저장한다.

    manual_search 결과가 없을 때 호출한다.
    수집 완료 후 manual_search를 다시 호출하여 답변을 생성한다.

    Args:
        brand: 세탁기 브랜드 (samsung)
        model: 세탁기 모델명 (예: WF-85A)

    Returns:
        "매뉴얼 수집을 완료했습니다. 다시 검색합니다." 또는 "매뉴얼을 찾을 수 없습니다."
    """
    try:
        success = await _get_crawler().crawl(brand, model)
        if success:
            return "매뉴얼 수집을 완료했습니다. 다시 검색합니다."
        return "매뉴얼을 찾을 수 없습니다."
    except NotImplementedError:
        # 6단계 구현 전 스텁 단계 방어
        return "매뉴얼을 찾을 수 없습니다."


# 4단계 StateGraph ToolNode에서 직접 사용
tools: list = [manual_search, manual_crawl]
