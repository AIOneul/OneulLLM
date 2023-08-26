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
    llm_topic = OpenAI(model_name='text-davinci-003', temperature=0.9)
    idea_prompt = f"""아래 기사를 1줄로 요약해줘
    ```
    {news}
    """
    main_idea = llm_topic(idea_prompt)
    print(main_idea)
    
    chat = ChatOpenAI(temperature=0.9) 
    template="아래 input을 바탕으로 토론의 시작 질문을 {input_level} 수준에서 {input_topic}로 정해줘."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="{text}"
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

