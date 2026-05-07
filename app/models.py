"""API 요청/응답 Pydantic 모델"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class CaseCreate(BaseModel):
    """케이스 생성 요청"""

    name: str
    description: str = ""


class Case(BaseModel):
    """케이스 정보"""

    id: str
    name: str
    description: str
    created_at: str
    status: Literal["open", "closed", "running", "done", "failed"]


class AnalysisRequest(BaseModel):
    """분석 시작 요청"""

    case_id: str
    disk_image_path: str
    prompt: str


class StrategyApproval(BaseModel):
    """전략 승인/수정 요청"""

    approved: bool
    feedback: str = ""


class PlanApproval(BaseModel):
    """계획 승인/수정 요청"""

    approved: bool
    feedback: str = ""


class StepUpdate(BaseModel):
    """Sub-Agent 단계 진행 이벤트"""

    step_index: int
    total: int
    step_name: str
    agent_name: str
    status: str
    output: str = ""
    elapsed: str = ""
    dfxml_fragment: str = ""
