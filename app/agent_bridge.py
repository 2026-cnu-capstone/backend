"""Agent 시스템 호출 브릿지

FastAPI 백엔드에서 agent src/ 모듈을 직접 import하여 호출
"""

from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

AGENT_ROOT = Path(__file__).parent.parent.parent / "agent"
AGENT_SRC = AGENT_ROOT / "src"
if str(AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(AGENT_SRC))

AGENT_ENV = AGENT_ROOT / ".env"
if AGENT_ENV.exists() and not os.environ.get("LLM_API_KEY"):
    from dotenv import load_dotenv
    load_dotenv(AGENT_ENV)

from agents.manager.graph import (
    ExecutionCallback,
    create_manager_state,
    run_execution,
    run_planning,
    run_report,
    run_strategy,
)
from config import LLMProvider, load_settings
from database.engine import get_engine, init_db
from llm_provider.base import BaseLLMProvider
from llm_provider.openai import OpenAIProvider
from llm_provider.anthropic import AnthropicProvider
from mcp_client.client import MCPClientManager
from rag.embedding import Embedder
from rag.pgvector_store import PgVectorStore
from rag.service import RAGService
from state.manager import ManagerState

from app.ws_manager import manager as ws_manager

MAX_SESSIONS = 50
"""최대 동시 분석 세션 수"""

_settings = None
_mcp: MCPClientManager | None = None
_llm_cache: dict[str, BaseLLMProvider] = {}
_rag_service: RAGService | None = None
_analysis_states: OrderedDict[str, ManagerState] = OrderedDict()
_cancel_flags: dict[str, bool] = {}


async def get_settings():
    """설정 로드 (캐시)"""
    global _settings
    if _settings is None:
        config_path = AGENT_SRC.parent / "config" / "mcp_servers.json"
        _settings = load_settings(config_path)
    return _settings


async def get_rag_service() -> RAGService | None:
    """RAG 서비스 인스턴스 반환 (설정 비활성 시 None)"""
    global _rag_service
    if _rag_service is not None:
        return _rag_service

    settings = await get_settings()
    if not settings.rag.enabled:
        return None

    engine = get_engine(settings.database_url)
    await init_db(engine)

    embedder = Embedder(settings.rag.embedding_model)
    store = PgVectorStore(engine, embedder)
    await store.ensure_extension()
    _rag_service = RAGService(
        store=store,
        top_k=settings.rag.search_top_k,
        similarity_threshold=settings.rag.similarity_threshold,
    )
    return _rag_service


async def get_llm(api_choice: str = "default") -> BaseLLMProvider:
    """LLM 프로바이더 인스턴스 반환 (api_choice별 캐시)"""
    if api_choice in _llm_cache:
        return _llm_cache[api_choice]

    settings = await get_settings()

    if api_choice == "mindlogic" and settings.mindlogic_api_key:
        from config import LLMConfig
        mindlogic_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model=settings.mindlogic_model,
            base_url=settings.mindlogic_base_url,
        )
        llm = OpenAIProvider(mindlogic_config, api_key=settings.mindlogic_api_key)
    elif settings.llm.provider == LLMProvider.OPENAI:
        llm = OpenAIProvider(settings.llm, api_key=settings.llm_api_key)
    else:
        llm = AnthropicProvider(settings.llm, api_key=settings.llm_api_key)

    _llm_cache[api_choice] = llm
    return llm


async def get_mcp() -> MCPClientManager:
    """MCP 클라이언트 매니저 인스턴스 반환 (싱글톤, 실패 시 재시도)"""
    global _mcp
    if _mcp is not None:
        return _mcp

    settings = await get_settings()
    mcp = MCPClientManager(settings.mcp)
    try:
        await mcp.connect_all()
    except Exception as exc:
        _mcp = None
        raise RuntimeError(f"MCP 서버 연결 실패: {exc}") from exc

    _mcp = mcp
    return _mcp


async def reset_mcp() -> None:
    """MCP 연결 초기화 (에러 복구용)"""
    global _mcp
    if _mcp is not None:
        try:
            await _mcp.disconnect_all()
        except Exception:
            pass
    _mcp = None


async def shutdown_mcp() -> None:
    """MCP 연결 종료 (앱 종료 시)"""
    global _mcp
    if _mcp is not None:
        await _mcp.disconnect_all()
        _mcp = None


def request_cancel(case_id: str) -> None:
    """실행 중단 요청"""
    _cancel_flags[case_id] = True


def is_cancelled(case_id: str) -> bool:
    """중단 요청 여부 확인"""
    return _cancel_flags.get(case_id, False)


def clear_cancel(case_id: str) -> None:
    """중단 플래그 초기화"""
    _cancel_flags.pop(case_id, None)


def cleanup_session(case_id: str) -> None:
    """분석 세션 정리"""
    _analysis_states.pop(case_id, None)
    _cancel_flags.pop(case_id, None)


def _enforce_session_limit() -> None:
    """세션 수 제한 초과 시 가장 오래된 세션 제거"""
    while len(_analysis_states) > MAX_SESSIONS:
        oldest_key = next(iter(_analysis_states))
        _analysis_states.pop(oldest_key)
        _cancel_flags.pop(oldest_key, None)


def get_session_status(case_id: str) -> dict[str, Any]:
    """분석 세션 현재 상태 조회"""
    state = _analysis_states.get(case_id)
    if not state:
        return {"exists": False}
    return {
        "exists": True,
        "phase": state.get("phase", "unknown"),
        "plan_steps_count": len(state.get("plan_steps", [])),
        "task_results_count": len(state.get("task_results", [])),
        "has_strategy": bool(state.get("analysis_strategy")),
        "has_plan": bool(state.get("plan_steps")),
    }


class WebSocketExecutionCallback:
    """Sub-Agent 실행 진행을 WebSocket으로 전송하는 콜백"""

    def __init__(self, case_id: str) -> None:
        self._case_id = case_id
        self._step_starts: dict[int, float] = {}

    def on_step_start(self, step_index: int, total: int, step: dict, agent_name: str) -> None:
        """단계 시작 이벤트"""
        import asyncio
        self._step_starts[step_index] = time.time()
        asyncio.create_task(
            ws_manager.send_event(self._case_id, "step_started", {
                "step_index": step_index,
                "total": total,
                "step_name": step.get("name", step.get("purpose", "")),
                "agent_name": agent_name,
            })
        )

    def on_step_done(self, step_index: int, total: int, step: dict, agent_name: str, result: dict) -> None:
        """단계 완료 이벤트"""
        import asyncio
        elapsed = time.time() - self._step_starts.get(step_index, time.time())
        event_data: dict[str, Any] = {
            "step_index": step_index,
            "total": total,
            "step_name": step.get("name", step.get("purpose", "")),
            "agent_name": agent_name,
            "status": result.get("status", "error"),
            "output": result.get("output", "")[:300],
            "elapsed": f"{int(elapsed)}s",
        }
        if result.get("dfxml_fragment"):
            event_data["dfxml_fragment"] = result["dfxml_fragment"]
        asyncio.create_task(
            ws_manager.send_event(self._case_id, "step_completed", event_data)
        )

    def on_step_skip(self, step_index: int, total: int, step: dict) -> None:
        """수동 단계 건너뛰기 이벤트"""
        import asyncio
        asyncio.create_task(
            ws_manager.send_event(self._case_id, "step_completed", {
                "step_index": step_index,
                "total": total,
                "step_name": step.get("name", ""),
                "agent_name": "manual",
                "status": "skip",
                "output": "",
                "elapsed": "0s",
            })
        )


async def start_analysis(case_id: str, disk_image_path: str, prompt: str) -> dict[str, Any]:
    """분석 시작: 시스템 프로필 추출 + 전략 생성"""
    _enforce_session_limit()
    mcp = await get_mcp()
    llm = await get_llm()
    rag = await get_rag_service()

    system_profile = ""
    try:
        result = await mcp.call_tool(
            "dissect__extract_system_profile",
            {"image_path": disk_image_path},
        )
        system_profile = mcp.get_tool_result_text(result)
    except Exception:
        pass

    state = create_manager_state(
        user_message=prompt,
        disk_image_path=disk_image_path,
        system_profile=system_profile,
    )

    state = await run_strategy(state, llm, rag_service=rag)
    _analysis_states[case_id] = state

    return {
        "strategy": state.get("analysis_strategy", ""),
        "system_profile": system_profile,
    }


async def approve_strategy(case_id: str, approved: bool, feedback: str = "") -> dict[str, Any]:
    """전략 승인 또는 수정 후 계획 생성"""
    state = _analysis_states.get(case_id)
    if not state:
        raise ValueError(f"분석 세션 없음: {case_id}")

    llm = await get_llm()
    mcp = await get_mcp()
    rag = await get_rag_service()

    if not approved and feedback:
        strategy = state.get("analysis_strategy", "")
        user_msg = state["messages"][0].get("content", "")
        state = create_manager_state(
            user_message=(
                f"{user_msg}\n\n[이전 전략]\n{strategy}\n\n"
                f"[수정 요청]: {feedback}\n위 수정 요청을 반영하여 새로운 분석 전략을 작성하세요."
            ),
            disk_image_path=state.get("disk_image_path"),
            disk_image_format=state.get("disk_image_format"),
            system_profile=state.get("system_profile"),
        )
        state = await run_strategy(state, llm, rag_service=rag)
        _analysis_states[case_id] = state
        return {
            "strategy": state.get("analysis_strategy", ""),
            "plan_ready": False,
        }

    state = await run_planning(state, llm, mcp, rag_service=rag)
    _analysis_states[case_id] = state

    steps = state.get("plan_steps", [])
    import logging
    logging.getLogger("agent_bridge").info(
        "plan_steps_sample: %s",
        [{k: s.get(k) for k in ("name", "mcp_server", "tool")} for s in steps[:3]],
    )

    return {
        "plan_text": state.get("analysis_plan", ""),
        "steps": steps,
        "plan_ready": True,
    }


async def approve_plan(case_id: str, approved: bool, feedback: str = "") -> dict[str, Any]:
    """계획 승인 또는 수정 후 재계획"""
    state = _analysis_states.get(case_id)
    if not state:
        raise ValueError(f"분석 세션 없음: {case_id}")

    llm = await get_llm()
    mcp = await get_mcp()
    rag = await get_rag_service()

    if not approved and feedback:
        plan = state.get("analysis_plan", "")
        user_msg = state["messages"][0].get("content", "")
        state = {
            **state,
            "messages": [{
                "role": "user",
                "content": (
                    f"{user_msg}\n\n[이전 계획]\n{plan}\n\n"
                    f"[수정 요청]: {feedback}\n위 수정 요청을 반영하여 새로운 계획을 작성하세요."
                ),
            }],
        }
        state = await run_planning(state, llm, mcp, rag_service=rag)
        _analysis_states[case_id] = state
        return {
            "plan_text": state.get("analysis_plan", ""),
            "steps": state.get("plan_steps", []),
        }

    _analysis_states[case_id] = state
    return {"approved": True}


async def execute_analysis(case_id: str) -> dict[str, Any]:
    """승인된 계획 실행"""
    state = _analysis_states.get(case_id)
    if not state:
        raise ValueError(f"분석 세션 없음: {case_id}")

    clear_cancel(case_id)
    llm = await get_llm()
    mcp = await get_mcp()
    rag = await get_rag_service()
    callback = WebSocketExecutionCallback(case_id)

    state = await run_execution(state, llm, mcp, callback=callback, rag_service=rag)
    _analysis_states[case_id] = state
    clear_cancel(case_id)

    return {
        "task_results": state.get("task_results", []),
        "evidence_repository": state.get("evidence_repository", []),
    }


async def generate_report(case_id: str) -> dict[str, str]:
    """보고서 생성"""
    state = _analysis_states.get(case_id)
    if not state:
        raise ValueError(f"분석 세션 없음: {case_id}")

    llm = await get_llm()
    result = await run_report(state, llm)
    cleanup_session(case_id)
    return result
