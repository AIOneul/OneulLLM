from fastapi import APIRouter, Body
from pydantic import BaseModel
from llm import llm_debate_start, llm_debate_feedback, llm_debate_response, llm_generate_quiz
from tools import parse_sentences, parse_data, remove_incomplete_sentences, extract_corrected_text
import config
import random
from typing import List
from typing import Tuple


router = APIRouter()

class DebateStart(BaseModel):
    level: int
    category: int
    news: str

class DebateResponse(BaseModel):
    level: int
    category: int
    logs: List[List[str]]

class Quiz(BaseModel):
    level: int
    news: List[str]
    
@router.post("/debate_start")
async def start_debate(debate: DebateStart):
    level_cat = config.LEVEL_INFO[debate.level]
    chosen_value = debate.category
    if chosen_value == 5000:
        random_values = [1000, 2000, 3000, 4000]
        chosen_value = random.choice(random_values)
    topic_cat = config.TOPIC_INFO[chosen_value]
    news = remove_incomplete_sentences(debate.news)
    print(news)
    response_from_llm = llm_debate_start(level_cat, topic_cat, news)
    return response_from_llm

@router.post("/debate_response")
async def response_debate(debate: DebateResponse):
    level_cat = config.LEVEL_INFO[debate.level]
    chosen_value = debate.category
    if chosen_value == 5000:
        random_values = [1000, 2000, 3000, 4000]
        chosen_value = random.choice(random_values)
    topic_cat = config.TOPIC_INFO[chosen_value]
    res = ""
    for log in debate.logs:
        res = log[1]
    print(res)
    response_from_llm = llm_debate_response(level_cat, topic_cat, res, debate.logs)
    feedback_from_llm = llm_debate_feedback(res)
    print(feedback_from_llm)
    feedback = extract_corrected_text(feedback_from_llm)
    print(feedback)
    if feedback is None:
        feedback = "올바른 문장"
    return response_from_llm, feedback

@router.post("/quiz")
async def generate_quiz(quiz: Quiz):
    level_cat = config.LEVEL_INFO[quiz.level]
    #sentences = parse_sentences(quiz.news, num_sentences=5)
    news_concatenated = '\n'.join(quiz.news)
    response_from_llm = llm_generate_quiz(level_cat, news_concatenated)
    return parse_data(response_from_llm)
