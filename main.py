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
import os, logging
os.environ["OPENAI_API_KEY"] = openai_api_key

logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("main class : ")


chat = ChatOpenAI(
    max_tokens=2048,
)


def generate_data(query,
                  data_path: str = database_path,
                  faiss_path: str = faiss_index_path):
    logger.info("Generating data for query: {}".format(query))

    logger.info("Fetching Google search results")
    google_search_results = get_google_search_results(query)
    logger.info("Google search results fetched successfully")

    logger.info("Fetching related questions from Google search results")
    goolge_related_questions = get_goolge_related_questions(google_search_results)
    logger.info("Related questions fetched successfully")

    logger.info("Fetching organic results from Google search results")
    goolge_organic_results = get_goolge_organic_results(google_search_results)
    logger.info(f"Organic results fetched successfully =={goolge_organic_results}")

    logger.info("Saving search content")
    save_search_content(query, data_path, goolge_related_questions, goolge_organic_results)
    logger.info("Search content saved successfully")




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    generate_data('Amazon company business model')
