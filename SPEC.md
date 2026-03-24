# QGuide Agent 재구현 스펙 (LangGraph v1)

## 개요

기존 프로젝트의 에이전트를 LangChain v1 / LangGraph 기반으로 재구현한다.
모바일 앱(React Native)의 백엔드 에이전트 역할이며, 에이전트 파트만 구현하고 FastAPI로 테스트한다.

---

## 확정 기술 스택

| 항목 | 결정 |
|---|---|
| LLM | EXAONE-3.5-7.8B (Ollama), 안 맞으면 교체 |
| Agent 구현 | LangGraph StateGraph |
| LangChain 버전 | v1 |
| 메모리 | SqliteSaver (롱텀, 재시작 후 유지) |
| LLM 캐싱 | SQLiteCache (동일 질문 LLM 재호출 방지) |
| 검색 방식 | 하이브리드 (ChromaDB 벡터 0.7 + Elasticsearch BM25 0.3) |
| 키워드 검색 | Elasticsearch (Nori 한국어 형태소 분석기) |
| Vector DB | ChromaDB |
| 이미지 인식 | 구현 범위 외 (brand, model 직접 입력으로 테스트) |
| 테스트 방식 | FastAPI 엔드포인트에 직접 입력값 전달 |
| 미들웨어 | 추후 필요 시 추가 |

---

## 폴더 구조

```
langgraph/
├── SPEC.md                  # 이 파일
├── requirements.txt         # 의존성
├── main.py                  # FastAPI 진입점
├── agent/
│   ├── __init__.py
│   ├── graph.py             # StateGraph 정의 (노드, 엣지, 조건)
│   ├── state.py             # State 구조 정의
│   └── tools.py             # manual_search, manual_crawl Tool 정의
├── search/
│   ├── __init__.py
│   ├── vector_search.py     # ChromaDB 벡터 검색
│   ├── keyword_search.py    # Elasticsearch BM25 키워드 검색
│   └── hybrid_search.py     # RRF 융합
├── store/
│   ├── __init__.py
│   ├── vector_store.py      # ChromaDB 연결 및 저장
│   └── embedding.py         # 임베딩 모델 (jhgan/ko-sroberta-multitask)
├── crawling/
│   ├── __init__.py
│   └── crawler.py           # 제조사별 크롤러 (온디맨드)
├── database/
│   ├── chroma_db/           # ChromaDB 저장 경로
│   └── conversations.db     # SqliteSaver 대화 기록
└── cache/
    └── .langchain_cache.db  # SQLiteCache LLM 응답 캐시
```

---

## State 구조

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 대화 히스토리 자동 누적
    brand: str                               # 세탁기 브랜드 (samsung / lg / daewoo)
    model: str                               # 세탁기 모델명 (예: SAM123X)
```

`brand`, `model`은 API 요청 시 직접 입력받아 초기 State에 주입한다.
사진 인식은 구현 범위 외이므로 항상 직접 입력값을 사용한다.

---

## Tool 인터페이스

### Tool 1 — `manual_search`

```
역할    : ChromaDB 하이브리드 검색 (벡터 + Elasticsearch BM25)
입력    : query(str), brand(str), model(str)
출력    : 검색 결과 텍스트 or "해당 제품의 매뉴얼이 등록되어 있지 않습니다."
호출 시점: 사용자 질문에 대해 항상 첫 번째로 호출
```

### Tool 2 — `manual_crawl`

```
역할    : 제조사 웹사이트에서 매뉴얼 크롤링 → ChromaDB 인덱싱
입력    : brand(str), model(str)
출력    : "매뉴얼 수집을 완료했습니다. 다시 검색합니다." or "매뉴얼을 찾을 수 없습니다."
호출 시점: manual_search 결과가 없을 때 LLM이 판단하여 호출
특이사항 : 크롤링 후 ChromaDB에 저장 → 이후 동일 모델 검색 시 크롤링 불필요
```

---

## StateGraph 설계

### 노드 구성

| 노드 | 역할 |
|---|---|
| `agent` | LLM이 메시지를 읽고 tool 호출 여부 결정 |
| `tools` | ToolNode — tool_calls를 실행 (manual_search or manual_crawl) |

### 엣지 구성

```
START → agent
agent → tools_condition → tools or END
tools → agent
```

### 조건 함수

```python
# langgraph.prebuilt의 tools_condition 사용
# tool_calls 있으면 "tools" 노드로, 없으면 END
from langgraph.prebuilt import tools_condition

graph_builder.add_conditional_edges("agent", tools_condition)
```

### 흐름 예시

```
사용자: "SAM123X 오류코드 E3 알려줘"
  ↓
agent 노드: manual_search("오류코드 E3", "samsung", "SAM123X") 호출 결정
  ↓
tools 노드: manual_search 실행 → "등록되어 있지 않습니다" 반환
  ↓
agent 노드: manual_crawl("samsung", "SAM123X") 호출 결정
  ↓
tools 노드: manual_crawl 실행 → 크롤링 + ChromaDB 저장 → "수집 완료" 반환
  ↓
agent 노드: manual_search("오류코드 E3", "samsung", "SAM123X") 재호출 결정
  ↓
tools 노드: manual_search 실행 → 검색 결과 반환
  ↓
agent 노드: tool_calls 없음 → 최종 답변 생성
  ↓
END
```

---

## 메모리 설정

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string("data/conversations.db")
app = graph.compile(checkpointer=memory)

# 사용자별 대화 분리
config = {"configurable": {"thread_id": "user_001"}}
app.invoke({"messages": [...], "brand": "samsung", "model": "SAM123X"}, config=config)
```

---

## LLM 캐싱 설정

```python
from langchain.cache import SQLiteCache
import langchain

langchain.llm_cache = SQLiteCache(database_path="cache/.langchain_cache.db")
```

동일한 (query + brand + model) 조합에 대해 LLM 재호출 없이 캐시 반환한다.

---

## 하이브리드 검색 설계

```
HybridSearchService
  ├── ChromaDB 벡터 검색       (가중치 0.7) — 의미 기반
  └── Elasticsearch BM25 검색  (가중치 0.3) — 키워드 기반 (모델명/오류코드 정확도)

Elasticsearch를 추가한 이유: SAM123X / SAM123Y처럼 비슷한 모델명이나
오류코드(E3, E4 등) 검색 시 벡터 검색만으로는 구분이 어렵기 때문
Nori 형태소 분석기로 한국어 도메인 용어(탈수, 헹굼 등) 정확 처리
```

---

## API 엔드포인트

### POST `/chat`

```json
Request:
{
  "query": "오류코드 E3 알려줘",
  "brand": "samsung",
  "model": "SAM123X",
  "thread_id": "user_001"
}

Response:
{
  "response": "SAM123X의 오류코드 E3는 ...",
  "thread_id": "user_001"
}
```

---

## 의존성 (requirements.txt)

```
langchain
langchain-community
langchain-core
langgraph
langchain-ollama
chromadb
elasticsearch
fastapi
uvicorn
sentence-transformers
beautifulsoup4
requests
```

---

## 구현 순서

```
1단계: 환경 설정 (requirements.txt, 폴더 구조 생성)
2단계: ChromaDB + 하이브리드 검색 (search/, database/)
3단계: Tool 정의 (agent/tools.py)
4단계: StateGraph 구성 (agent/state.py, agent/graph.py)
5단계: FastAPI 연결 (main.py)
6단계: 크롤러 연결 (crawling/crawler.py)
7단계: 테스트 및 검증
```
