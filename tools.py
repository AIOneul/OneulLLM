import random
import re

def parse_sentences(input_text, num_sentences=5):
    sentences = input_text.split('.')
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]  # Remove empty sentences

    if len(sentences) <= num_sentences:
        return sentences
    
    selected_sentences = random.sample(sentences, num_sentences)
    return selected_sentences

def parse_data(data):
    questions = re.findall(r"질문: (.*?)\n", data)
    options = re.findall(r"\n\s*가\) (.*?)\n\s*나\) (.*?)\n\s*다\) (.*?)\n", data)
    answers = re.findall(r"정답: (.*?)\n", data)

    parsed_data = []

    for q, opt, a in zip(questions, options, answers):
        parsed_data.append({
            "question": q,
            "options": {
                "가": opt[0],
                "나": opt[1],
                "다": opt[2]
            },
            "answer": a
        })

    return parsed_data

def remove_incomplete_sentences(paragraph):
    # Split sentences based on space after punctuation
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    
    # Filter out incomplete sentences (those not ending in ., ?, ! or those ending in .. or ...)
    complete_sentences = [s for s in sentences if s.endswith(('.', '?', '!')) and not s.endswith(('..', '...'))]
    
    # Join the sentences back together
    return ' '.join(complete_sentences)

def extract_corrected_text(feedback: str) -> str:
    """Extract the corrected text from feedback string."""
    # Check if "고친 문장:" exists in the feedback
    if "고친 문장:" in feedback:
        # Split the string based on "고친 문장:" and return the text after it
        return feedback.split("고친 문장:")[1].strip()
    return None  # return original feedback if "고친 문장:" is not present

