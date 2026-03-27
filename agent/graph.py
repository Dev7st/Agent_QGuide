import os

import langchain
from langchain_community.cache import SQLiteCache
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.state import AgentState
from agent.tools import tools

# LLM 캐시 설정 — 동일 질문에 대한 LLM 재호출 방지 (모듈 임포트 시 1회 실행)
langchain.llm_cache = SQLiteCache(
    database_path=os.getenv("LLM_CACHE_PATH")
)

# LLM 초기화 — EXAONE-3.5-7.8B, tool_calls 지원 모델
_llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL"),
    base_url=os.getenv("OLLAMA_URL"),
)

# LLM에 tools 바인딩 — LLM이 tool 스키마를 인식하여 tool_calls 결정
_llm_with_tools = _llm.bind_tools(tools)


async def _agent_node(state: AgentState) -> dict:
    """Agent 노드 — LLM이 messages를 읽고 tool_calls 여부를 결정"""
    response = await _llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


# StateGraph 조립
_graph_builder = StateGraph(AgentState)
_graph_builder.add_node("agent", _agent_node)
_graph_builder.add_node("tools", ToolNode(tools))

_graph_builder.add_edge(START, "agent")

# tools_condition — tool_calls 있으면 "tools" 노드로, 없으면 END
_graph_builder.add_conditional_edges("agent", tools_condition)
_graph_builder.add_edge("tools", "agent")

# SqliteSaver — 재시작 후에도 대화 히스토리 유지 (thread_id 단위로 분리)
_memory = SqliteSaver.from_conn_string(
    os.getenv("MEMORY_DB_PATH")
)

# 그래프 컴파일 — 모듈 레벨 싱글턴, 임포트 시 1회만 실행
app = _graph_builder.compile(checkpointer=_memory)


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
    result = await app.ainvoke(initial_state, config=config)
    return result["messages"][-1].content
