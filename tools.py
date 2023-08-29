import random
import re

def parse_sentences(input_text, num_sentences=5):
    """
    Split the input_text into sentences and return a random subset.
    """
    sentences = [s.strip() for s in input_text.split('.') if s.strip()]
    return sentences if len(sentences) <= num_sentences else random.sample(sentences, num_sentences)

def parse_data(data):
    """
    Extract questions, options, and answers from a structured data string.
    """
    questions = re.findall(r"질문: (.*?)\n", data)
    options = re.findall(r"\n\s*가\) (.*?)\n\s*나\) (.*?)\n\s*다\) (.*?)\n", data)
    answers = re.findall(r"정답: (.*?)\n", data)

    return [
        {
            "question": q,
            "options": {
                "가": opt[0],
                "나": opt[1],
                "다": opt[2]
            },
            "answer": a
        }
        for q, opt, a in zip(questions, options, answers)
    ]

def remove_incomplete_sentences(paragraph):
    """
    Remove sentences that don't end in standard punctuation or those that end with '..' or '...'.
    """
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    complete_sentences = [
        s for s in sentences if s.endswith(('.', '?', '!')) and not s.endswith(('..', '...'))
    ]
    return ' '.join(complete_sentences)

def extract_corrected_text(feedback: str) -> str:
    """
    Extract the corrected text portion from a feedback string, if it exists.
    """
    if "고친 문장:" in feedback:
        return feedback.split("고친 문장:")[1].strip()
    return None

def compare_strings(s1, s2):
    words1 = s1.split()
    words2 = s2.split()

    i, j = 0, 0
    result = []
    while i < len(words1) or j < len(words2):
        if i < len(words1) and j < len(words2) and words1[i] == words2[j]:
            same_words1, same_words2 = "", ""
            while i < len(words1) and j < len(words2) and words1[i] == words2[j]:
                same_words1 += words1[i] + " "
                same_words2 += words2[j] + " "
                i += 1
                j += 1
            result.append([same_words1.strip(), same_words2.strip()])
        else:
            diff_word1, diff_word2 = "", ""
            k, l = i, j
            while k < len(words1):
                diff_word1 += words1[k] + " "
                k += 1
                l = j
                found = False
                while l < len(words2):
                    if k < len(words1) and l < len(words2) and words1[k] == words2[l]:
                        found = True
                        break
                    diff_word2 += words2[l] + " "
                    l += 1
                if found:
                    break
            i = k
            j = l
            result.append([diff_word1.strip(), diff_word2.strip()])

    return result