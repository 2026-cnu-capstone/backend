import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.models import Case, CaseCreate

router = APIRouter()

# In-memory store
_cases: dict[str, Case] = {}


@router.get("/cases", response_model=list[Case])
def get_cases():
    return list(_cases.values())


@router.get("/cases/{case_id}", response_model=Case)
def get_case(case_id: str):
    case = _cases.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@router.post("/cases", response_model=Case, status_code=201)
def create_case(data: CaseCreate):
    case = Case(
        id=str(uuid.uuid4()),
        name=data.name,
        description=data.description,
        created_at=datetime.now(timezone.utc).isoformat(),
        status="open",
    )
    _cases[case.id] = case
    return case
