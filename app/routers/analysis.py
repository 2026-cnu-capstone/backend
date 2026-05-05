"""분석 워크플로우 API 라우터"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.agent_bridge import (
    approve_plan,
    approve_strategy,
    execute_analysis,
    generate_report,
    get_session_status,
    request_cancel,
    start_analysis,
)
from app.models import AnalysisRequest, PlanApproval, StrategyApproval
from app.ws_manager import manager as ws_manager

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


@router.post("/start")
async def api_start_analysis(req: AnalysisRequest):
    """분석 시작 — 시스템 프로필 추출 + 전략 생성"""
    try:
        result = await start_analysis(req.case_id, req.disk_image_path, req.prompt)
        await ws_manager.send_event(req.case_id, "strategy_ready", result)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{case_id}/strategy/approve")
async def api_approve_strategy(case_id: str, req: StrategyApproval):
    """전략 승인/수정"""
    try:
        result = await approve_strategy(case_id, req.approved, req.feedback)
        if result.get("plan_ready"):
            await ws_manager.send_event(case_id, "plan_ready", result)
        else:
            await ws_manager.send_event(case_id, "strategy_ready", result)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{case_id}/plan/approve")
async def api_approve_plan(case_id: str, req: PlanApproval):
    """계획 승인/수정"""
    try:
        result = await approve_plan(case_id, req.approved, req.feedback)
        if not req.approved:
            await ws_manager.send_event(case_id, "plan_ready", result)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{case_id}/execute")
async def api_execute_analysis(case_id: str):
    """승인된 계획 실행 (WebSocket으로 진행상황 push)"""
    try:
        result = await execute_analysis(case_id)
        await ws_manager.send_event(case_id, "execution_done", {
            "task_results": result["task_results"],
        })
        return {"status": "done", "total_steps": len(result["task_results"])}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        await ws_manager.send_event(case_id, "error", {"message": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{case_id}/pause")
async def api_pause_analysis(case_id: str):
    """실행 중단 요청"""
    request_cancel(case_id)
    return {"status": "paused"}


@router.post("/{case_id}/report")
async def api_generate_report(case_id: str):
    """보고서 생성"""
    try:
        result = await generate_report(case_id)
        await ws_manager.send_event(case_id, "report_ready", result)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{case_id}/status")
async def api_get_status(case_id: str):
    """현재 분석 상태 조회"""
    return get_session_status(case_id)


@router.websocket("/ws/{case_id}")
async def websocket_analysis(websocket: WebSocket, case_id: str):
    """실시간 분석 이벤트 WebSocket"""
    await ws_manager.connect(case_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(case_id, websocket)
    except Exception:
        await ws_manager.disconnect(case_id, websocket)
