from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationKGMemory
import os
import config
os.environ["OPENAI_API_KEY"] = config.API_KEYS["OPENAI_API_KEY"]

def llm_debate_start(level, topic, news):
    
    topic_chat = ChatOpenAI(temperature=0.9) 
    topic_template="""
    ### 지시 ###
    아래 신문 기사의 핵심 단어를 1개 제공해줘
    
    [예시]
    기사:음주운전으로 이미 5차례나 처벌받았음에도 또다시 음주사고를 낸 뒤 운전자까지 바꿔치기한 50대 남성이 실형을 선고받고 법정 구속됐습니다.
    핵심 단어:음주운전
    """
    topic_system_message_prompt = SystemMessagePromptTemplate.from_template(topic_template)
    topic_human_template="""
    ```
    기사:{text}
    핵심 단어:"""
    topic_human_message_prompt = HumanMessagePromptTemplate.from_template(topic_human_template)
    topic_chat_prompt = ChatPromptTemplate.from_messages([topic_system_message_prompt, topic_human_message_prompt])
    topic_chatchain = LLMChain(llm=topic_chat, prompt=topic_chat_prompt)
    
    main_idea = topic_chatchain.run(input_level= level, input_topic=topic, text=news)
    print(main_idea)
    
    chat = ChatOpenAI(temperature=0.9) 
    template="""
    당신은 {input_level}을 가르치는 선생님입니다. 
    오늘은 학생들의 토론의 진행 역할을 하고 있습니다. 
    핵심 단어를 바탕으로 {input_topic}에 대한 토론의 시작 질문을 {input_level}이 알아들을 수 있도록 제공해주세요.
    
    [예시]
    핵심 단어:
    질문:

    [예시]
    핵심 단어:
    질문:
    
    [예시]
    핵심 단어:
    질문:
    
    [예시]
    핵심 단어:
    질문:
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="핵심 단어:{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chatchain = LLMChain(llm=chat, prompt=chat_prompt)
    return chatchain.run(input_level= level, input_topic=topic, text=main_idea)

def llm_debate_response(level, topic, res):
    chat = ChatOpenAI(temperature=0.9) 
    template="상대방의 대답에 맞추어 적절한 답변을 해서 토론을 이어가 주세요.  {input_level} 수준에서 {input_topic}에 대해서."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chatchain = LLMChain(llm=chat, prompt=chat_prompt)
    return chatchain.run(input_level= level, input_topic=topic, text=res)

def llm_debate_feedback(res):
    chat = ChatOpenAI(temperature=0.9) 
    template="문맥, 맞춤법, 적절한 어휘 사용을 검토해서 고친 문장을 줘"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chatchain = LLMChain(llm=chat, prompt=chat_prompt)
    return chatchain.run(text=res)

def llm_generate_quiz(level, topic, news):
    chat = ChatOpenAI(temperature=0.9) 
    template="""
    너는 초등학교 선생님이야. 너는 입력받은 기사에서 초등학생 어휘력 향상을 위한 퀴즈를 만드는 일을 할거야. 입력받은 기사 예시에서 어휘의 의미를 물어보는 질문을 만들어줘
    [조건]
    1. 퀴즈는 각 문장당 하나씩, 총 5개의 퀴즈를 만들어줘
    2. 3지 선다 문제를 만들어줘
    3. {input_level} 수준의 문제를 만들어줘
    4.정답을 함께 추출해줘
    
    ```
    [예시]
    Q:
    몸체를 비스듬히 기울여 달에 접근한 찬드라얀 3호가 선체를 바로 세우고 착륙 준비를 합니다.달 표면에 무사히 닿자 연구진들 사이에서 기립박수와 환호가 터져 나옵니다.물을 분해해 수소를 연료로 쓰거나 산소, 식수도 공급할 수 있어 지구에서 물자를 조달할 시간과 에너지를 대폭 줄일 수 있기 때문입니다.인도의 성공 불과 사흘 전엔 먼저 러시아가 무인 달 탐사선 '루나 25호'를 발사했지만, 달 표면에 추락하며 반세기 만의 도전이 실패로 끝났습니다.
    
    A:
    질문:'찬드라얀 3호가 선체를 바로 세우고 착륙 준비를 한다'에서 "착륙"의 의미는 무엇일까요?
    가) 물건을 떨어뜨리다
    나) 땅에 닿아 멈추다
    다) 물을 마시다
    정답: 나) 땅에 닿아 멈추다
    
    질문:기사에서 "물을 분해해 수소를 연료로 쓰거나 산소, 식수도 공급할 수 있다"라고 나오는데, "분해"의 뜻은 무엇일까요?
    가) 물건을 합치다
    나) 물건을 나누다
    다) 물건을 바꾸다
    정답: 나) 물건을 나누다

    질문:기사에 따르면 "무인 달 탐사선 '루나 25호'는 달 표면에 추락하며 실패로 끝났다"라고 나와있습니다. "추락"의 의미는 무엇일까요?
    가) 물체가 높은 곳으로 올라가다
    나) 물체가 땅으로 떨어지다
    다) 물체가 움직이지 않다
    정답: 나) 물체가 땅으로 떨어지다
    
    질문:기사에서 "연구진들 사이에서 기립박수와 환호가 터져 나옵니다."에서 "기립박수"의 의미는 무엇일까요?
    가) 서 있는 상태로 박수를 치다.
    나) 땅에 누워 박수를 치다.
    다) 손을 흔들며 박수를 치다.
    정답: 가) 서 있는 상태로 박수를 치다.
    
    질문:기사에서 "연구진들 사이에서 기립박수와 환호가 터져 나옵니다."에서
    가) 웃음을 지으며 소리를 내다.
    나) 기뻐하며 춤을 추다.
    다) 큰 소리로 함성을 지르다.
    정답: 다) 큰 소리로 함성을 지르다.
    
    """
    print("질문이왔어요!")
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="""
    Q:
    {text}
    A:"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chatchain = LLMChain(llm=chat, prompt=chat_prompt)
    return chatchain.run(input_level= level, text=news)
