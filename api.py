from fastapi import APIRouter, Body
from pydantic import BaseModel
from llm import llm_debate_start

router = APIRouter()

class DebateStart(BaseModel):
    level: int
    topic: int
    news: str

class DebateResponse(BaseModel):
    level: int
    topic: str
    res: str

@router.post("/debate_start/")
async def start_debate(debate: DebateStart):
    # Call the llm_debate_start function
    response_from_llm = llm_debate_start(debate.level, debate.topic, debate.news)
    return response_from_llm

@router.post("/debate_start/")
async def start_debate(debate: DebateResponse):
    return {"level": debate.level, "topic": debate.topic, "Response": debate.res}
