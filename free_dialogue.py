import os
import langchain
from config import *
from util import *
from langchain.llms import OpenAI, Cohere, HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from typing import Optional, Type
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, Tool, initialize_agent, tool
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

prompt = PromptTemplate(
    input_variables=["text"],
    template="{text}",
)

llm = OpenAI(temperature=0)
chat = ChatOpenAI(temperature=0)

llm_chain = LLMChain(llm=llm, prompt=prompt)
chat_model_chain = LLMChain(llm=chat, prompt=prompt)


class CustomSearchTool(BaseTool):
    name = "search tool"
    description = "一个搜索引擎。 当您你需要回答有关时事的问题调用该工具，否则不要使用该工具。 输入应该是搜索查询。"

    def _run(self, query: str) -> str:
        """Use the tool."""
        return search(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


# You can create the tool to pass to an agent
chat_tool = Tool(
    name="Chat",
    description="一个非常有用的助理，可以回答常见的问题，比如写文章、规划行程、等等",
    func=chat_model_chain.run,
    return_direct=True

)


def get_free_dialogue_answer(query):
    tools = [chat_tool, CustomSearchTool()]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    result = agent.run(query)
    return result


if __name__ == '__main__':
    query = "农夫山泉的市值"
    get_free_dialogue_answer(query)
