import re
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import httpx
from bs4 import BeautifulSoup
from elasticsearch.helpers import async_bulk

from search.keyword_search import ES_INDEX, KeywordSearchService
from store.embedding import EmbeddingService
from store.vector_store import VectorStore

# 이전 프로젝트(samsung_crawler.py)에서 실제 동작 확인된 API 엔드포인트
SAMSUNG_MODEL_URL = "https://www.samsung.com/sec/support/model/"
SAMSUNG_MANUAL_API = "https://www.samsung.com/sec/xhr/goods/goodsManual"
SAMSUNG_US_MODEL_URL = "https://www.samsung.com/us/support/"

# goodsId 추출 정규식 패턴 — 이전 프로젝트(samsung_crawler.py)의 6개 패턴
GOODS_ID_PATTERNS = [
    r'goodsId\s*:\s*["\']([G]\d+)["\']',
    r'"goodsId"\s*:\s*"([G]\d+)"',
    r"'goodsId'\s*:\s*'([G]\d+)'",
    r'data-goods-id=["\']([G]\d+)["\']',
    r'goodsId=([G]\d+)',
    r'"goodsId":"([G]\d+)"',
]

# 삼성 매뉴얼 페이지 푸터 패턴 — 페이지 경계 감지용
# 왼쪽 페이지: "2 한국어", 오른쪽 페이지: "한국어 3"
PAGE_FOOTER_PATTERN = re.compile(
    r'^\d+\s+한국어$|^한국어\s+\d+$',
    re.MULTILINE,
)

# 청킹 크기 기준
SECTION_MAX_SIZE = 500   # 이 이상이면 세분화
CHUNK_TARGET_SIZE = 400  # 세분화 시 목표 크기
CHUNK_MIN_SIZE = 100     # 최소 청크 크기


class ManualCrawler:
    """제조사 웹사이트에서 매뉴얼을 크롤링하여 ChromaDB + Elasticsearch에 저장하는 크롤러

    Note:
        삼성(samsung) 크롤러만 지원한다.
        의존성 주입 미제공 시 내부에서 직접 생성한다.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        keyword_search: Optional[KeywordSearchService] = None,
    ) -> None:
        self._embedding = embedding_service or EmbeddingService()
        self._vector_store = vector_store or VectorStore()
        self._keyword_search = keyword_search or KeywordSearchService()

    # ──────────────────────────────────────────────────────────
    # 메인 진입점
    # ──────────────────────────────────────────────────────────

    async def crawl(self, brand: str, model: str) -> bool:
        """제조사 사이트에서 매뉴얼을 크롤링하여 ChromaDB + ES에 저장한다.

        Args:
            brand: 세탁기 브랜드 (samsung)
            model: 세탁기 모델명 (예: WF-85A)

        Returns:
            True  — 크롤링 성공 및 저장 완료
            False — 매뉴얼을 찾을 수 없거나 처리 실패
        """
        try:
            # 미지원 브랜드 조기 종료
            if brand != "samsung":
                print(f"[ManualCrawler] 미지원 브랜드: {brand}")
                return False

            # 1. 매뉴얼 URL 탐색
            url = await self._find_samsung_manual_url(model)
            if not url:
                print(f"[ManualCrawler] 매뉴얼 URL 탐색 실패: {model}")
                return False

            # 2. 파일 다운로드
            result = await self._download(url)
            if not result:
                print(f"[ManualCrawler] 다운로드 실패: {url}")
                return False
            content_bytes, content_type = result

            # 3. 텍스트 추출 (PDF / HTML 분기)
            if "pdf" in content_type.lower():
                raw_text = self._extract_text_from_pdf(content_bytes)
            else:
                raw_text = self._extract_text_from_html(content_bytes)

            if not raw_text.strip():
                print(f"[ManualCrawler] 텍스트 추출 실패 또는 빈 문서: {url}")
                return False

            # 4. 청킹
            documents = self._chunk_text(raw_text, brand, model)
            if not documents:
                print(f"[ManualCrawler] 청킹 결과 없음: {model}")
                return False

            # 5. ChromaDB 저장
            await self._store_to_chroma(documents)

            # 6. ES 인덱싱 (실패해도 True 유지 — 벡터 검색은 ChromaDB로 동작)
            try:
                await self._index_to_es(documents)
            except Exception as e:
                print(f"[ManualCrawler] ES 인덱싱 실패 (무시): {e}")

            print(f"[ManualCrawler] 크롤링 완료: {brand} {model} ({len(documents)}청크)")
            return True

        except Exception as e:
            print(f"[ManualCrawler] 크롤링 예외 발생: {e}")
            return False

    # ──────────────────────────────────────────────────────────
    # Samsung 매뉴얼 URL 탐색 (3단계 전략)
    # ──────────────────────────────────────────────────────────

    async def _find_samsung_manual_url(self, model: str) -> Optional[str]:
        """Samsung 매뉴얼 PDF URL을 3단계 전략으로 탐색한다.

        공통: 모델 페이지 HTML 1회 요청
        1단계: CSS 셀렉터 + JS 정규식으로 PDF 링크 직접 탐색
        2단계: goodsId 추출 → 내부 매뉴얼 API POST (동적 로딩 대응)
        3단계: Fallback — 미국 사이트 시도

        Returns:
            PDF URL 문자열 또는 None
        """
        model_url = f"{SAMSUNG_MODEL_URL}{model}/"

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # 공통: 모델 페이지 HTML 1회 요청
            try:
                response = await client.get(model_url, headers=self._browser_headers())
                response.raise_for_status()
                html_content = response.text
            except Exception as e:
                print(f"[ManualCrawler] 모델 페이지 요청 실패: {e}")
                return None

            # 1단계: CSS 셀렉터 + JS 소스 정규식으로 PDF 링크 직접 탐색
            pdf_url = self._find_pdf_link_in_html(html_content)
            if pdf_url:
                return pdf_url

            # 2단계: goodsId 추출 → 매뉴얼 API POST (PDF가 동적 로딩될 때)
            goods_id: Optional[str] = None
            for pattern in GOODS_ID_PATTERNS:
                match = re.search(pattern, html_content)
                if match:
                    goods_id = match.group(1)
                    break

            if goods_id:
                pdf_url = await self._fetch_manual_via_api(client, model, goods_id)
                if pdf_url:
                    return pdf_url

            # 3단계: Fallback — 미국 사이트 시도
            us_url = f"{SAMSUNG_US_MODEL_URL}{model}/"
            try:
                us_response = await client.get(us_url, headers=self._browser_headers())
                us_response.raise_for_status()
                pdf_url = self._find_pdf_link_in_html(us_response.text)
                if pdf_url:
                    return pdf_url
            except Exception:
                pass

        return None

    async def _fetch_manual_via_api(
        self,
        client: httpx.AsyncClient,
        model: str,
        goods_id: str,
    ) -> Optional[str]:
        """Samsung 내부 매뉴얼 API에 POST 요청하여 PDF URL을 추출한다."""
        try:
            headers = {
                **self._browser_headers(),
                "Referer": f"{SAMSUNG_MODEL_URL}{model}/",
                "X-Requested-With": "XMLHttpRequest",
            }
            data = {
                "mdlCode": model,
                "goodsId": goods_id,
                "manualLang": "KO",
                "supportYn": "Y",
            }
            response = await client.post(SAMSUNG_MANUAL_API, data=data, headers=headers)
            response.raise_for_status()
            return self._find_pdf_link_in_html(response.text)
        except Exception as e:
            print(f"[ManualCrawler] 매뉴얼 API 요청 실패: {e}")
            return None

    def _find_pdf_link_in_html(self, html: str) -> Optional[str]:
        """HTML에서 PDF 링크를 CSS 셀렉터 + 정규식으로 추출한다.

        삼성 실제 DOM 구조 기준:
            <a class="btn-download" data-nmfile="*.pdf" href="https://downloadcenter.samsung.com/...pdf?_gl=...">
        href가 쿼리 파라미터를 포함하므로 a[href$=".pdf"]는 동작하지 않는다.
        """
        soup = BeautifulSoup(html, "html.parser")

        # CSS 셀렉터 순서대로 시도 (삼성 실제 DOM 기준)
        selectors = [
            'a.btn-download',             # 삼성 다운로드 버튼 클래스 (실제 확인)
            'a[data-nmfile$=".pdf"]',      # data-nmfile 속성으로 PDF 식별
            '.list-type-download a',       # 매뉴얼 목록 컨테이너 하위 링크
        ]
        for selector in selectors:
            tag = soup.select_one(selector)
            if tag and tag.get("href"):
                href = str(tag["href"])
                # 상대 경로를 절대 경로로 변환
                if href.startswith("//"):
                    return f"https:{href}"
                elif href.startswith("/"):
                    return f"https://www.samsung.com{href}"
                return href

        # 정규식 fallback — JS 소스 포함 전체 HTML에서 PDF URL 탐색
        # 쿼리 파라미터(?_gl=...) 포함 URL도 캡처
        match = re.search(r'https?://[^\s"\'<>]+\.pdf(?:\?[^\s"\'<>]*)?', html)
        if match:
            return match.group(0)

        return None

    # ──────────────────────────────────────────────────────────
    # HTTP 다운로드
    # ──────────────────────────────────────────────────────────

    async def _download(self, url: str) -> Optional[tuple[bytes, str]]:
        """URL에서 파일을 다운로드하여 (content_bytes, content_type)을 반환한다.

        Returns:
            (bytes, content_type) 또는 None (HTTP 오류 / 타임아웃 시)
        """
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(url, headers=self._browser_headers())
                response.raise_for_status()
                content_type = response.headers.get("content-type", "application/octet-stream")
                return response.content, content_type
        except httpx.TimeoutException as e:
            print(f"[ManualCrawler] 다운로드 타임아웃: {e}")
            return None
        except httpx.HTTPStatusError as e:
            print(f"[ManualCrawler] HTTP 오류 {e.response.status_code}: {url}")
            return None
        except Exception as e:
            print(f"[ManualCrawler] 다운로드 예외: {e}")
            return None

    # ──────────────────────────────────────────────────────────
    # 텍스트 추출
    # ──────────────────────────────────────────────────────────

    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """PyMuPDF(fitz)로 텍스트 기반 PDF에서 텍스트를 추출한다.

        이미지 기반 PDF(스캔본)는 지원하지 않는다.

        Returns:
            추출된 텍스트 (이미지 기반 PDF이면 빈 문자열)
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            texts: List[str] = []
            for page in doc:
                text = page.get_text()
                if text.strip():
                    texts.append(text)
            doc.close()
            return "\n".join(texts)
        except Exception as e:
            print(f"[ManualCrawler] PDF 파싱 실패: {e}")
            return ""

    def _extract_text_from_html(self, html_bytes: bytes) -> str:
        """BeautifulSoup으로 HTML에서 본문 텍스트를 추출한다.

        nav, header, footer, script, style 태그를 제거하여 노이즈를 줄인다.
        """
        soup = BeautifulSoup(html_bytes, "html.parser")
        for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n")

    # ──────────────────────────────────────────────────────────
    # 텍스트 청킹 (HYBRID_SECTION_SIZE 전략)
    # ──────────────────────────────────────────────────────────

    def _chunk_text(
        self, text: str, brand: str, model: str
    ) -> List[Dict[str, Any]]:
        """페이지 푸터 기준 1차 분리 후 크기 기준 세분화로 청킹한다.

        - 페이지 푸터("2 한국어" / "한국어 3")로 페이지 단위 분리
        - 500자 이하 페이지 → 1개 청크 유지
        - 500자 초과 페이지 → 400자 목표로 문장 경계 세분화
        - 최소 청크 크기: 100자

        Returns:
            [{"id": ..., "text": ..., "metadata": ...}, ...]
        """
        # 페이지 푸터 위치로 텍스트 분할
        sections = self._split_by_page_footers(text)

        chunks: List[str] = []
        for section in sections:
            section = section.strip()
            if not section:
                continue

            if len(section) <= SECTION_MAX_SIZE:
                # 500자 이하 → 그대로 1청크
                chunks.append(section)
            else:
                # 500자 초과 → 문장 경계로 세분화
                chunks.extend(self._split_by_sentences(section))

        # 최소 크기 필터링 및 문서 구조 생성
        documents: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            if len(chunk) >= CHUNK_MIN_SIZE:
                documents.append({
                    "id": f"{brand}_{model}_{i:04d}",
                    "text": chunk,
                    "metadata": {"brand": brand, "model": model},
                })

        return documents

    def _split_by_page_footers(self, text: str) -> List[str]:
        """페이지 푸터 위치를 기준으로 텍스트를 페이지 단위로 분할한다.

        푸터 자체("2 한국어" 등)는 청크에서 제거한다.
        푸터가 없으면 전체를 하나의 페이지로 처리한다.
        """
        matches = list(PAGE_FOOTER_PATTERN.finditer(text))
        if not matches:
            return [text]

        pages: List[str] = []
        prev_end = 0

        for match in matches:
            # 푸터 이전 텍스트를 한 페이지로
            page_text = text[prev_end:match.start()].strip()
            if page_text:
                pages.append(page_text)
            prev_end = match.end()

        # 마지막 푸터 이후 남은 텍스트
        tail = text[prev_end:].strip()
        if tail:
            pages.append(tail)

        return pages

    def _split_by_sentences(self, text: str) -> List[str]:
        """문장 경계(. ! ?)를 기준으로 CHUNK_TARGET_SIZE 단위로 세분화한다."""
        # 문장 분리 — 마침표, 느낌표, 물음표 뒤 공백 또는 줄바꿈 기준
        sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_pattern.split(text)

        chunks: List[str] = []
        current_parts: List[str] = [] # 현재 청크에 담긴 문장들
        current_len = 0 # 현재 담긴 총 글자 수

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_len + sentence_len > CHUNK_TARGET_SIZE and current_parts:
                chunks.append(" ".join(current_parts))
                current_parts = [sentence]
                current_len = sentence_len
            else:
                current_parts.append(sentence)
                current_len += sentence_len

        if current_parts:
            chunks.append(" ".join(current_parts))

        return chunks

    # ──────────────────────────────────────────────────────────
    # ChromaDB 저장
    # ──────────────────────────────────────────────────────────

    async def _store_to_chroma(self, documents: List[Dict[str, Any]]) -> None:
        """임베딩을 생성하고 ChromaDB에 저장한다.

        VectorStore.add_manual_embeddings() 내부에서 _remove_existing_manual()이
        호출되므로 중복 처리는 불필요하다.
        """
        texts = [d["text"] for d in documents]
        embeddings = await self._embedding.embed_texts(texts)
        self._vector_store.add_manual_embeddings(documents, embeddings)

    # ──────────────────────────────────────────────────────────
    # Elasticsearch 인덱싱
    # ──────────────────────────────────────────────────────────

    async def _index_to_es(self, documents: List[Dict[str, Any]]) -> None:
        """해당 brand/model 문서만 ES에 직접 인덱싱한다.

        migrate_from_vector_store() 미사용 — 전체 재인덱싱 비용 방지.
        실패 시 crawl()에서 예외를 캐치하므로 여기서는 그냥 raise.
        """
        await self._keyword_search._ensure_index()

        actions = [
            {
                "_index": ES_INDEX,
                "_id": doc["id"],
                "_source": {
                    "text": doc["text"],
                    "brand": doc["metadata"]["brand"],
                    "model": doc["metadata"]["model"],
                },
            }
            for doc in documents
        ]
        await async_bulk(self._keyword_search._es, actions)

    # ──────────────────────────────────────────────────────────
    # 유틸리티
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _browser_headers() -> Dict[str, str]:
        """크롤링 차단 방지용 브라우저 헤더"""
        return {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
