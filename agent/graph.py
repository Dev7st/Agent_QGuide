import os

import langchain
from langchain_community.cache import SQLiteCache
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.state import AgentState
from agent.tools import tools

# LLM 캐시 설정 — 동일 질문에 대한 LLM 재호출 방지 (모듈 임포트 시 1회 실행)
langchain.llm_cache = SQLiteCache(
    database_path=os.getenv("LLM_CACHE_PATH")
)

# LLM 초기화 — tool_calls 지원 모델, thinking 모드 비활성화
_llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL"),
    base_url=os.getenv("OLLAMA_URL"),
    think=False,  # thinking 모드 비활성화 — 불필요한 추론 과정 제거
)

# LLM에 tools 바인딩 — LLM이 tool 스키마를 인식하여 tool_calls 결정
_llm_with_tools = _llm.bind_tools(tools)


async def _agent_node(state: AgentState) -> dict:
    """Agent 노드 — LLM이 messages를 읽고 tool_calls 여부를 결정"""
    # brand/model을 SystemMessage로 주입 — tool 호출 시 LLM이 올바른 값을 사용하도록
    system = SystemMessage(
        content=(
            f"세탁기 브랜드: {state['brand']}, 모델명: {state['model']}.\n"
            "tool 호출 시 반드시 이 brand와 model 값을 사용하라.\n"
            "반드시 다음 순서를 따르라:\n"
            "1. manual_search를 먼저 호출한다.\n"
            "2. manual_search 결과가 없으면 반드시 manual_crawl을 호출한다.\n"
            "3. manual_crawl 완료 후 manual_search를 다시 호출하여 답변한다.\n"
            "매뉴얼 없이 임의로 답변하지 말라.\n"
            "검색된 내용 중 질문과 직접 관련된 부분만 사용하라. 관련 없는 내용은 포함하지 말라."
        )
    )
    response = await _llm_with_tools.ainvoke([system] + state["messages"])
    return {"messages": [response]}


# StateGraph 조립
_graph_builder = StateGraph(AgentState)
_graph_builder.add_node("agent", _agent_node)
_graph_builder.add_node("tools", ToolNode(tools))

_graph_builder.add_edge(START, "agent")

# tools_condition — tool_calls 있으면 "tools" 노드로, 없으면 END
_graph_builder.add_conditional_edges("agent", tools_condition)
_graph_builder.add_edge("tools", "agent")

async def run(query: str, brand: str, model: str, thread_id: str) -> str:
    """그래프 실행 진입점 — 5단계 main.py에서 호출

    Args:
        query: 사용자 질문
        brand: 세탁기 브랜드 (samsung)
        model: 세탁기 모델명 (예: WF-85A)
        thread_id: 사용자/세션 단위 대화 분리 키

    Returns:
        LLM 최종 답변 텍스트
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "brand": brand,
        "model": model,
    }
    # AsyncSqliteSaver — 비동기 환경에서 대화 히스토리 유지 (thread_id 단위로 분리)
    async with AsyncSqliteSaver.from_conn_string(os.getenv("MEMORY_DB_PATH")) as memory:
        app = _graph_builder.compile(checkpointer=memory)
        result = await app.ainvoke(initial_state, config=config)
    return result["messages"][-1].content
