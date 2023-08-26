from fastapi import APIRouter, Body
from pydantic import BaseModel
from llm import llm_debate_start, llm_debate_feedback, llm_debate_response
import config
import random

router = APIRouter()

class DebateStart(BaseModel):
    level: int
    topic: int
    news: str

class DebateResponse(BaseModel):
    level: int
    topic: int
    res: str

@router.post("/debate_start/")
async def start_debate(debate: DebateStart):
    level_cat = config.LEVEL_INFO[debate.level]
    chosen_value = debate.topic
    if chosen_value == 5000:
        random_values = [1000, 2000, 3000, 4000]
        chosen_value = random.choice(random_values)
    topic_cat = config.TOPIC_INFO[chosen_value]
    
    response_from_llm = llm_debate_start(level_cat, topic_cat, debate.news)
    return response_from_llm

@router.post("/debate_response/")
async def response_debate(debate: DebateResponse):
    level_cat = config.LEVEL_INFO[debate.level]
    chosen_value = debate.topic
    if chosen_value == 5000:
        random_values = [1000, 2000, 3000, 4000]
        chosen_value = random.choice(random_values)
    topic_cat = config.TOPIC_INFO[chosen_value]
    
    response_from_llm = llm_debate_response(level_cat, topic_cat, debate.res)
    feedback_from_llm = llm_debate_feedback(debate.res)
    return response_from_llm, feedback_from_llm