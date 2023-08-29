from typing import List, Tuple
import logging

# Third-party imports
from fastapi import APIRouter, Body
from pydantic import BaseModel

# Local imports
from llm import (
    llm_debate_start,
    llm_debate_feedback,
    llm_debate_response,
    llm_generate_quiz,
)
from tools import (
    parse_sentences,
    parse_data,
    remove_incomplete_sentences,
    extract_corrected_text,
    compare_strings
)
import config

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


def choose_topic_category(category: int) -> int:
    """Choose a topic category, potentially picking randomly based on given category."""
    if category == 5000:
        return random.choice([1000, 2000, 3000, 4000])
    return category


@router.post("/debate_start")
async def start_debate(debate: DebateStart):
    print(f"Request payload: {debate.dict()}")
    
    level_cat = config.LEVEL_INFO[debate.level]
    chosen_value = choose_topic_category(debate.category)
    topic_cat = config.TOPIC_INFO[chosen_value]
    news = remove_incomplete_sentences(debate.news)
    response = llm_debate_start(level_cat, topic_cat, news)

    print(f"Response: {response}")
    return response

@router.post("/debate_response")
async def response_debate(debate: DebateResponse):
    print(f"Request payload: {debate.dict()}")
    
    level_cat = config.LEVEL_INFO[debate.level]
    chosen_value = choose_topic_category(debate.category)
    topic_cat = config.TOPIC_INFO[chosen_value]
    
    last_log = debate.logs[0][1]
    response_from_llm = llm_debate_response(level_cat, topic_cat, last_log, debate.logs)
    
    feedback_from_llm = llm_debate_feedback(last_log)
    feedback = extract_corrected_text(feedback_from_llm)
    print(feedback)
    if feedback:
        dict_feedback = compare_strings(last_log, feedback)
    else:
        dict_feedback = [[last_log, last_log]]
    final_response = {
        "res": response_from_llm,
        "feedback": dict_feedback
    }
    
    print(f"Response: {final_response}")
    return final_response

@router.post("/quiz")
async def generate_quiz(quiz: Quiz):
    print(f"Request payload: {quiz.dict()}")
    
    level_cat = config.LEVEL_INFO[quiz.level]
    news_concatenated = '\n'.join(quiz.news)
    response_from_llm = llm_generate_quiz(level_cat, news_concatenated)
    print(response_from_llm)
    parsed_data = parse_data(response_from_llm)

    print(f"Response: {parsed_data}")
    return parsed_data
