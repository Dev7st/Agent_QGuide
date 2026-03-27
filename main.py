from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.graph import run

app = FastAPI(title="QGuide Agent API")


class ChatRequest(BaseModel):
    query: str       # 사용자 질문
    brand: str       # 세탁기 브랜드 (samsung)
    model: str       # 세탁기 모델명 (예: SAM123X)
    thread_id: str   # 사용자/세션 단위 대화 분리 키


class ChatResponse(BaseModel):
    response: str    # LLM 최종 답변
    thread_id: str   # 요청에서 받은 thread_id 그대로 반환


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """에이전트 호출 엔드포인트 — StateGraph를 실행하고 LLM 최종 답변을 반환"""
    try:
        answer = await run(request.query, request.brand, request.model, request.thread_id)
    except Exception:
        raise HTTPException(status_code=500, detail="에이전트 오류가 발생했습니다.")
    return ChatResponse(response=answer, thread_id=request.thread_id)


@app.get("/health")
async def health() -> dict:
    """헬스체크 — 서버 기동 확인용"""
    return {"status": "ok"}
