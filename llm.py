from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationKGMemory
import os
import config
os.environ["OPENAI_API_KEY"] = config.API_KEYS["OPENAI_API_KEY"]

def llm_debate_start(level, topic, news):
    llm = OpenAI(model_name="text-davinci-003", temperature=0.9)
    resp = llm("안녕")
    print(resp)
    return resp
    """
    Generate a debate start statement based on given parameters.

    Parameters:
    - level: The level of the debate (e.g., "beginner", "intermediate", "advanced").
    - topic: The topic for the debate.
    - string_format: A string format to customize the output.

    Returns:
    - A formatted string to start the debate.
    """
    '''
    llm = OpenAI(temperature=0.9)
    
    template = "~~탬플릿 내용~~"
    
    prompt = PromptTemplate(
        input_variables=["history", "input"], 
        template=template
    )

    memory = ConversationKGMemory(llm=llm)
    memory.save_context({"input":"5만원 이하 네이비 크롭 니트 찾아줘"}, {"output":"https://www.musinsa.com/categories/item/001006?color=36&price1=0&price2=50000&includeKeywords=크롭"})
    memory.save_context({"input":"40,000원 이하 크롭 블랙 니트를 찾아줘"}, {"output":"https://www.musinsa.com/categories/item/001006?color=2&price1=0&price2=40000&includeKeywords=크롭"})

    conversation_with_kg = ConversationChain(
        llm=llm,
        verbose=True,
        prompt=prompt,
        memory=memory
    )

    return conversation_with_kg.predict(input=user_input)
    '''
