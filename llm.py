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
from langchain.memory import ConversationBufferMemory
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
    핵심 단어를 바탕으로 {input_topic}에 대한 토론의 시작 질문을 {input_level}이 알아들을 수 있도록 물어봐주세요.
    
    [예시]
    핵심 단어: 후쿠시마 오염수 방수
    질문: 후쿠시마 오염수 방출이 일본의 국내 경제와 국제 경제에 어떠한 영향을 미칠 것이라고 생각하십니까?

    [예시]
    핵심 단어: 인공지능 일자리
    질문: 인공지능이 우리의 미래 일자리에 어떤 영향을 미칠 것이라고 생각하나요?
    
    [예시]
    핵심 단어: 환경 보호
    질문: 환경을 잘 보호하려면 우리가 어떤 행동을 해야 할까요?
    
    [예시]
    핵심 단어:전동 킥보드
    질문: 전동 킥보드를 사용할 때 핼멧 착용을 의무화하는 것에 대한 여러분의 생각은 어떠한가요
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="""핵심 단어:{text}
                    질문:"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chatchain = LLMChain(llm=chat, prompt=chat_prompt)
    return chatchain.run(input_level= level, input_topic=topic, text=main_idea)

def llm_debate_response(level, topic, res, logs):
    
    memory = ConversationBufferMemory()
    for log in logs:  
        memory.save_context({"input": log[0]}, {"output": log[1]})
    print(memory.load_memory_variables({}))
    llm = ChatOpenAI(temperature=0.9) 
    
    input_message = f"""당신은 {level}이다. 현재 {level}과 {topic}을 주제로 논리적인 토론을 하고 있다.
    토론의 발제는 다음과 같다. {logs[0][0]}
    당신은 상대방의 대답에 맞추어 3문장 정도로 {level}이 이해할 수 있는 적절한 답변을 해서 토론을 이어가 주세요.
    우선 공감하는 표현의 말을 한마디 합니다. 첫번째 문장은 상대의 말을 요약합니다. 두번째 문장은 상대의 의견에 반박합니다. 세번째 문장은 상대에게 질문을 합니다.
    
    ```
    상대방의 답변: {res}

    """
    conversation = ConversationChain(
        llm=llm, 
        memory=memory,
        #prompt=chat_prompt
    )
    return conversation.predict(input= input_message)
    
def llm_debate_feedback(res):
    chat = ChatOpenAI(temperature=0.9) 
    template="""당신은 국어 선생님입니다. 적절한 국어 사용을 피드백해주는 것에 능합니다.
    문맥, 맞춤법, 적절한 어휘 사용을 검토해서 고친 문장을 줘.
    고칠 필요가 없을 경우 "적절한 문장이에요"를 줘.
    
    [예시]
    그 책은 나의 탁자가 있어.
    A:"탁자가 있어"의 "가"는 적절하지 않아. 고친문장: 그 책은 나의 탁자에 있어.
    우리는 비가 오는 날에 밖에 나가서 축구를 했댜.
    A:"했댜"는 맞춤법 오류야. 고친문장: 우리는 비가 오는 날에 밖에 나가서 축구를 했다.
    고양이가 박스 안테로 넣었어.
    나는 친구와 같이 영화를 보러 영화관에 갔다.
    A: 적절한 문장이에요.
    A:"안테로 넣었어"의 "테로"는 적절하지 않아. 고친문장: 고양이를 박스 안에 넣었어.
    학교는 오후 2시에 퇴근하다.
    A:"학교가 퇴근하다"는 주어와 동사가 적절한 조합이 아니야. 고친문장: 학교는 오후 2시에 끝났다.
    너의 지갑이 어디에 넣었냐?
    A: "지갑이 어디에 넣었냐"의 "이"는 적절하지 않아. 고친문장: 너의 지갑을 어디에 넣었어?
    신발가게에서 신발을 샀어.
    A: 적절한 문장이에요.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="""```
    {text}
    A:"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chatchain = LLMChain(llm=chat, prompt=chat_prompt)
    return chatchain.run(text=res)

def llm_generate_quiz(level, news):
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
    몸체를 비스듬히 기울여 달에 접근한 찬드라얀 3호가 선체를 바로 세우고 착륙 준비를 합니다.
    긴장 최고조의 순간.
    달 표면에 무사히 닿자 연구진들 사이에서 기립박수와 환호가 터져 나옵니다.
    브릭스 정상회의 참석차 화상으로 착륙을 지켜 본 나렌드라 모디 총리도 얼굴에 미소를 띄운 채 국기를 흔듭니다.
    현지시간 8월 23일 오후 6시 4분, 인도가 인류 역사상 최초로 달 남극 착륙에 성공했습니다.
    그간 달의 남극은 많은 물이 얼음 상태로 존재할 가능성이 높아서 심우주 진출의 교두보로 여겨져 왔습니다.
    물을 분해해 수소를 연료로 쓰거나 산소, 식수도 공급할 수 있어 지구에서 물자를 조달할 시간과 에너지를 대폭 줄일 수 있기 때문입니다.
    앞으로 찬드라얀 3호는 달 남극에 얼음과 각종 광물 등이 있는지 확인하는 작업을 수행합니다.
    지난 2019년 실패 이후 두 번째 시도 만에 쾌거를 이룬 가운데, 달 남극을 향한 우주 강국들의 경쟁도 보다 심화할 전망입니다.
    인도의 성공 불과 사흘 전엔 먼저 러시아가 무인 달 탐사선 '루나 25호'를 발사했지만, 달 표면에 추락하며 반세기 만의 도전이 실패로 끝났습니다.
    내년엔 중국의 '창어' 6, 7호가 출격을 준비하고 있고, 내후년엔 미국이 우주비행사들을 달의 남극으로 보낼 계획입니다.A:
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
    
    질문:달에 도착한 찬드라얀 3호는 어떤 행동을 하려고 하나요?
    가) 선체를 세우고 착륙하기
    나) 물을 분해해 연료로 사용하기
    다) 우주비행사들을 보내기
    정답: 가) 선체를 세우고 착륙하기

    달의 남극에 무엇이 존재할 가능성이 높아서 연구되고 있는 걸까요?
    가) 눈
    나) 얼음
    다) 바위
    정답: 나) 얼음 
    """
    
    print("질문이왔어요!")
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="""
    ```
    {text}
    A:"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chatchain = LLMChain(llm=chat, prompt=chat_prompt)
    return chatchain.run(input_level= level, text=news)
