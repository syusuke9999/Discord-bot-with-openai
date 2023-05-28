from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import GoogleSearchAPIWrapper
import datetime
import pytz

jst = pytz.timezone('Asia/Tokyo')


def initialize_agent(user_key: str = None):
    search = GoogleSearchAPIWrapper()
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
        )
    ]
    datetime_jst = datetime.datetime.now(jst)
    now = datetime_jst
    now_of_year = now.strftime("%Y")
    now_of_month = now.strftime("%m")
    now_of_day = now.strftime("%d")
    now_of_time = now.strftime("%H:%M")
    prefix = """Today is the year {now_of_year}, the month is {now_of_month} and the date {now_of_day}.
    The current time is {now_of_time}. 
    You are a Discord bot residing in a Discord channel for people interested in Discord bots that work with OpenAI's API\". 
    Please have a conversation with users about how \"Discord bots that work with OpenAI's API\" can be useful to them.
    Avoid mentioning the topic of the prompt and greet them considering the current time. 
    Don't use English, please communicate only in Japanese.
    Also, please keep your replies to 450 tokens or less.
    You have access to the following tools:
    """.format(now_of_year=now_of_year, now_of_month=now_of_month, now_of_day=now_of_day, now_of_time=now_of_time)
    suffix = """Begin!
    {chat_history}
    Question: {input}
    {agent_scratchpad}"""
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory)
    return agent_chain
