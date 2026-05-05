"""WebSocket 연결 관리"""

from __future__ import annotations

import json
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    """케이스별 WebSocket 연결 관리"""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, case_id: str, websocket: WebSocket) -> None:
        """새 WebSocket 연결 등록"""
        await websocket.accept()
        if case_id not in self._connections:
            self._connections[case_id] = []
        self._connections[case_id].append(websocket)

    def disconnect(self, case_id: str, websocket: WebSocket) -> None:
        """WebSocket 연결 해제"""
        if case_id in self._connections:
            self._connections[case_id] = [
                ws for ws in self._connections[case_id] if ws != websocket
            ]
            if not self._connections[case_id]:
                del self._connections[case_id]

    async def send_event(self, case_id: str, event_type: str, data: dict[str, Any]) -> None:
        """특정 케이스의 모든 연결에 이벤트 전송"""
        message = json.dumps({"type": event_type, **data}, ensure_ascii=False)
        connections = self._connections.get(case_id, [])
        disconnected: list[WebSocket] = []

        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.disconnect(case_id, ws)


manager = ConnectionManager()
