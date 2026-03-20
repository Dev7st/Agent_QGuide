# QGuide Agent 재구현 — Claude 컨텍스트

## 프로젝트 배경

기존 QGuide 프로젝트(세탁기 매뉴얼 AI 어시스턴트)의 에이전트 파트를 LangChain v1 / LangGraph 기반으로 재구현한다.
기존 프로젝트의 문제(AgentExecutor 미사용, 5단계 불필요한 래핑, 중복 로직)를 개선하는 것이 목적이다.
모바일 앱(React Native) 백엔드용이지만, 에이전트 파트만 구현하고 FastAPI로 테스트한다.
이미지 인식(사진 → 모델명 추출)은 구현 범위 외이므로 brand, model은 항상 직접 입력한다.

## 확정 기술 스택

| 항목 | 결정 |
|---|---|
| LLM | EXAONE-3.5-7.8B (Ollama), 안 맞으면 교체 |
| Agent 구현 | LangGraph StateGraph |
| LangChain 버전 | v1 |
| 메모리 | SqliteSaver (롱텀, 재시작 후 유지) |
| LLM 캐싱 | SQLiteCache (동일 질문 LLM 재호출 방지) |
| 검색 방식 | 하이브리드 (ChromaDB 벡터 0.7 + BM25 키워드 0.3) |
| Vector DB | ChromaDB |
| Tool 1 | manual_search — ChromaDB 하이브리드 검색 |
| Tool 2 | manual_crawl — 온디맨드 크롤링 (검색 실패 시 LLM이 호출) |
| 미들웨어 | 추후 필요 시 추가 |

## 폴더 구조

상세 내용은 [SPEC.md](SPEC.md) 참고

```
langgraph/
├── CLAUDE.md
├── SPEC.md
├── RULES.md
├── .claude/
│   └── rules.md
├── requirements.txt
├── main.py
├── agent/
│   ├── graph.py
│   ├── state.py
│   └── tools.py
├── search/
│   ├── vector_search.py
│   ├── keyword_search.py
│   └── hybrid_search.py
├── store/
│   ├── vector_store.py
│   └── embedding.py
├── crawling/
│   └── crawler.py
├── database/
└── cache/
```

## StateGraph 핵심 흐름

```
START → agent → should_continue() → tools → agent → END
                                  ↘ END (tool_calls 없으면 종료)
```

tool calling 방식 사용 — LLM이 직접 tool 호출 여부와 순서를 결정한다.
ReAct 방식 사용하지 않는다.

## 구현 순서

```
1단계: 환경 설정
2단계: ChromaDB + 하이브리드 검색
3단계: Tool 정의
4단계: StateGraph 구성
5단계: FastAPI 연결
6단계: 크롤러 연결
7단계: 테스트 및 검증
```

## 참고 문서

- [SPEC.md](SPEC.md) — 아키텍처 상세 스펙 (State 구조, Tool 인터페이스, API 설계)
- [RULES.md](RULES.md) — 커밋 메시지, 타입 힌트, 환경변수 규칙
- [.claude/rules.md](.claude/rules.md) — Claude 행동 규칙
