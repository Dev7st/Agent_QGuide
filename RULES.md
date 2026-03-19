# 개발 규칙

## 커밋 메시지 타입

| 타입 | 설명 | 예시 |
|---|---|---|
| `feat` | 새 기능 구현 | `feat: manual_search tool 구현` |
| `fix` | 버그 수정 | `fix: BM25 검색 필터 오류 수정` |
| `docs` | 문서 수정 | `docs: SPEC.md 업데이트` |
| `refactor` | 기능 변경 없는 코드 개선 | `refactor: hybrid_search 단순화` |

## 타입 힌트

모든 함수에 입력과 반환 타입을 명시한다.

```python
# good
def manual_search(query: str, brand: str, model: str) -> str:

# bad
def manual_search(query, brand, model):
```

## 환경변수

API 키, 경로 등 민감한 값은 `.env`에 저장하고 코드에 직접 작성하지 않는다.

```python
# good
import os
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# bad
OLLAMA_URL = "http://localhost:11434"
```
