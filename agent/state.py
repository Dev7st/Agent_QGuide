from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """StateGraph 전체에서 공유되는 상태 — 노드 간 메시지와 제품 정보를 전달"""

    # 대화 히스토리 — add_messages 리듀서가 새 메시지를 덮어쓰지 않고 누적
    messages: Annotated[list, add_messages]

    # 세탁기 브랜드 (samsung) — graph 호출 시 최초 주입, 이후 불변
    brand: str

    # 세탁기 모델명 (예: WF-85A) — graph 호출 시 최초 주입, 이후 불변
    model: str
