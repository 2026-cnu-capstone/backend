"""FastAPI 애플리케이션 엔트리포인트"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent_bridge import shutdown_mcp
from app.routers import cases
from app.routers.analysis import router as analysis_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 MCP 연결 관리"""
    yield
    await shutdown_mcp()


app = FastAPI(title="Forensic Multi-Agent Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cases.router, prefix="/api")
app.include_router(analysis_router)
