# Claude 행동 규칙

## 코드 작성 규칙

- 모든 함수에 타입 힌트를 반드시 포함한다
- 주석은 한국어로 작성한다
- 민감한 값(.env)은 코드에 직접 작성하지 않는다

## 구현 원칙

- 불필요한 래핑 금지 — 기존 프로젝트의 5단계 래핑 문제를 반복하지 않는다
- 단계별 승인 후 진행 — 각 구현 단계마다 확인 후 다음 단계로 넘어간다
- SPEC.md 기준 준수 — 확정된 스펙 외 기능을 임의로 추가하지 않는다
- LangChain v1 기준 — 0.x 버전 API 사용하지 않는다

## 기존 프로젝트 반복 금지

다음은 기존 프로젝트의 문제점이다. 재구현 시 반복하지 않는다.

- AgentExecutor import 후 미사용
- RetrieverFactory, HybridRetriever 등 불필요한 래핑 레이어
- _fallback_vector_search() — VectorSearchService와 중복된 로직
- vector_cache와 llm_cache 이중 관리 — SQLiteCache 하나로 충분

## 커밋 명령어 제안 규칙

커밋 명령어를 제안할 때 RULES.md의 타입을 따른다.
- feat / fix / docs / refactor 중 하나를 prefix로 사용한다