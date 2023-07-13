import langchain
from serpapi import GoogleSearch
import json
import logging
import pandas as pd
import os
import openai
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
import re
from config import *
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def get_google_search_results(query):
    logger.info(f"start get_google_search_results input is {query}")
    try:
        params = {
            "q": query,
            "gl": "cn",
            "hl": "zh-cn",
            "num": 5,
            "no_cache": True,
            "api_key": SERPAPI_API_KEY
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        # results = json.dumps(results,ensure_ascii=False)
        logger.info(f"get_google_sear5ch_results out is(result) {results}")
        return results

    except Exception as e:
        logger.warning(f"An error occurred while getting the search results: {e}")
        logger.warning(f"get_google_sear5ch_results out is(query) {query}")
        return query


def get_goolge_organic_results(json_data):
    # 解析Google search的结果： get_goolge_organic_results
    logger.info("start get_goolge_organic_results")
    try:
        result_strings = []
        # 解析organic_results字段
        if 'organic_results' in json_data:
            organic_results = json_data['organic_results']

            # 解析每个搜索结果
            for result in organic_results:
                # 解析title字段
                if 'title' in result:
                    title_text = result['title']
                else:
                    title_text = ""
                # 解析snippet字段
                if 'snippet' in result:
                    snippet_text = result['snippet']
                else:
                    snippet_text = ""
                # 解析about_this_result字段
                if 'about_this_result' in result:
                    about_result = result['about_this_result']
                    about_keywords = ""
                    # 解析keywords字段
                    if 'keywords' in about_result:
                        about_keywords = about_result['keywords']
                        about_keywords = " ".join(about_keywords)
                    # 拼接结果字符串
                    # result_string = f"Title: {title_text}\nKeywords: {about_keywords}\nSnippet: {snippet_text}\n"
                    result_string = f"{title_text}\n{about_keywords}\n{snippet_text}\n"
                else:
                    # 拼接结果字符串
                    # result_string = f"Title: {title_text}\nSnippet: {snippet_text}\n"
                    result_string = f"{title_text}\n{snippet_text}\n"
                result_strings.append(result_string)
            # 将所有结果字符串拼接为一个字符串并返回
        if len(result_strings) > 0:
            result_strings = ["\n".join(result_strings[i:i + 3]) for i in range(0, len(result_strings), 3)]
        return result_strings[0:1]
    except Exception as e:
        logger.info("start get_goolge_organic_results somthing wrong")
        return []


def search(query):
    res = get_google_search_results(query)
    if 'error' in str(res)[:40]:
        logger.info("something wrong with serper api")
    if "answer_box" in res:
        return str(res["answer_box"])[0:1600]
    else:
        return str(get_goolge_organic_results(res))[0:1600]


def check_taiwan(sentence):
    sentence = str(sentence).lower().replace(" ", "").replace("\n", "")
    pattern = r"(台.{0,4}湾|台.{0,4}弯|tai.{0,4}wan)"
    match = re.search(pattern, sentence, flags=re.IGNORECASE)
    if match:
        return 1
    return 0


def check_who(sentence):
    sentence = str(sentence).lower().replace(" ", "").replace("\n", "")
    pattern = r"(你.{0,3}谁|你叫.{0,4}什么|who.{0,4}you|what.{0,5}name)"
    match = re.search(pattern, sentence, flags=re.IGNORECASE)
    if match:
        return 1
    return 0
