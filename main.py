import langchain
from serpapi import GoogleSearch
import json
from logging import getLogger
import pandas as pd
import os
import openai
from langchain.llms import AzureOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from config import *
from util import summarize, get_google_search_results, get_goolge_related_questions
from util import get_goolge_organic_results, save_search_content, similarity_search

logger = getLogger()
openai.api_key = openai_api_key

chat = ChatOpenAI(
    max_tokens=2048,
)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
