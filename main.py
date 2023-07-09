import langchain
from langchain.chat_models import ChatOpenAI
from util import summarize, get_google_search_results, get_goolge_related_questions
from util import get_goolge_organic_results, save_search_content, similarity_search
import os, logging, json, fire
import pandas as pd
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from config import *

logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

os.environ["OPENAI_API_KEY"] = openai_api_key
chat = ChatOpenAI(
)

if __name__ == '__main__':
    fire.Fire(main)

