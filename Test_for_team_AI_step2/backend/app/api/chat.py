from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from ..db.session import SessionLocal
from ..services.chat_service import answer_query

router = APIRouter()

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)

class AskResponse(BaseModel):
    answer: str
    matches: list

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, db: Session = Depends(get_db)):
    return answer_query(db, req.query, req.top_k)
