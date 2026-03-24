class ManualCrawler:
    """제조사 웹사이트에서 매뉴얼을 크롤링하여 ChromaDB에 저장하는 크롤러

    Note:
        실제 크롤링 로직은 6단계에서 구현한다.
        삼성(samsung) 크롤러만 지원한다.
    """

    async def crawl(self, brand: str, model: str) -> bool:
        """제조사 사이트에서 매뉴얼을 크롤링하여 ChromaDB에 저장한다.

        Args:
            brand: 세탁기 브랜드 (samsung)
            model: 세탁기 모델명 (예: WF-85A)

        Returns:
            True  — 크롤링 성공 및 ChromaDB 저장 완료
            False — 매뉴얼을 찾을 수 없음
        """
        raise NotImplementedError("6단계에서 구현 예정")
