from fastapi import APIRouter, Body
from pydantic import BaseModel
from llm import llm_debate_start, llm_debate_feedback, llm_debate_response, llm_generate_quiz
from tools import parse_sentences, parse_data, remove_incomplete_sentences
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

class Quiz(BaseModel):
    level: int
    topic: int
    news: str
    
@router.post("/debate_start/")
async def start_debate(debate: DebateStart):
    level_cat = config.LEVEL_INFO[debate.level]
    chosen_value = debate.topic
    if chosen_value == 5000:
        random_values = [1000, 2000, 3000, 4000]
        chosen_value = random.choice(random_values)
    topic_cat = config.TOPIC_INFO[chosen_value]
    news = remove_incomplete_sentences(debate.news)
    print(news)
    response_from_llm = llm_debate_start(level_cat, topic_cat, news)
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

@router.post("/quiz/")
async def generate_quiz(quiz: Quiz):
    level_cat = config.LEVEL_INFO[quiz.level]
    chosen_value = quiz.topic
    if chosen_value == 5000:
        random_values = [1000, 2000, 3000, 4000]
        chosen_value = random.choice(random_values)
    topic_cat = config.TOPIC_INFO[chosen_value]
    #sentences = parse_sentences(quiz.news, num_sentences=5)
    response_from_llm = llm_generate_quiz(level_cat, topic_cat, quiz.news)
    return parse_data(response_from_llm)
