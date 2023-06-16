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
import os

os.environ["OPENAI_API_KEY"] = openai_api_key
embeddings = OpenAIEmbeddings()
chat = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k"
)

# from temp_model import *
# chat = chat
# embeddings = embeddings

logger = getLogger()


def summarize(text, summary_prompt):
    # 根据search的document 生成 summary
    logger.info(f"start summarize input is {text}")
    try:
        messages = [
            SystemMessage(content=summary_prompt),
            HumanMessage(content=text)
        ]
        result = chat(messages)
        logger.info(f" summarize output is (result.content):  {result.content}")
        return result.content
    except Exception as e:
        logger.warning("summarize class something wrong")
        logger.warning(f"An error occurred: {str(e)}")
        logger.warning(f" summarize output is (text): {text}")
        return text


def get_google_search_results(query):
    logger.info(f"start get_google_search_results input is {query}")
    try:
        params = {
            "q": query,
            "location": "Austin, Texas",
            "api_key": google_search_api_key
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


def get_goolge_related_questions(data):
    # 解析Google search的结果： get_goolge_related_questions
    logger.info("start get_goolge_related_questions")
    related_questions_res = []
    try:
        # 检查related_questions字段是否存在
        if 'related_questions' in data:
            related_questions = data['related_questions']
            # 解析related_questions中的每个问题
            for question in related_questions:
                # 检查question字段是否存在
                if 'question' in question:
                    question_text = question['question']
                else:
                    snippet_text = ""
                # 检查snippet字段是否存在
                if 'snippet' in question:
                    snippet_text = question['snippet']
                else:
                    snippet_text = ""
                # 检查title字段是否存在
                if 'title' in question:
                    title_text = question['title']
                else:
                    title_text = ""
                # 检查list字段是否存在
                if 'list' in question:
                    list_items = question['list']
                    list_items = " ".join(list_items)
                else:
                    list_items = ""
                # 打印解析结果

                sub_related_questions = f"Question: {question_text}\nTitle: {title_text}\nSnippet: {snippet_text}\nItems: {list_items}\n"
                # sub_related_questions=str(question_text)+"\n"+str(title_text)+"\n"+str(snippet_text)+"\n"+str(list_items)
                related_questions_res.append(sub_related_questions)
                # print(sub_related_questions)
                # print("\n\n")
        return related_questions_res
    except Exception as e:
        logger.info("start get_goolge_related_questions somthing wrong")
        return []


def save_search_content(query, data_path, related_questions, organic_results, faiss_path):
    try:
        logger.info("start save_search_content")
        logger.info(f"organic_results: {organic_results}")

        summary = [summarize(query + "\n" + str(i), summary_prompt) for i in organic_results]

        logger.info("summary again")
        summary = [summarize(i, summary_prompt_2) for i in summary]

        organic_results_dict = {"query": query, "content_type": "organic_results", "content": organic_results,
                                "summary": summary}
        df2 = pd.DataFrame.from_dict(organic_results_dict)

        if os.path.isfile(data_path):
            logger.info("Merging content with existing data")
            data = pd.read_csv(data_path)
            df2 = pd.concat([data, df2])
        logger.info("Saving df2 to data_path")
        df2.to_csv(data_path, index=False, encoding="utf_8")

        # 取消索引
        # logger.info(f"start save search content index, creating it...")
        # loader = CSVLoader(file_path=data_path, encoding="utf_8")
        # docs = loader.load()
        # db = FAISS.from_documents(docs, embeddings)
        # db.save_local(faiss_path)
        # logger.info(f"Index created and saved successfully.")
        # logger.info(f"Index path is : {faiss_path}")

        # return df2
        return "\n".join(summary)
    except Exception as e:
        logger.error(f"save_search_content An error occurred: {str(e)}")
        logger.error(f"just return query : {query}")
        return query


def similarity_search(faiss_path, data_path, query):
    logger.info(f"start similarity_search, input is {query}")
    try:
        if not os.path.exists(faiss_path):
            logger.info(f"Index does not exist, creating it...")
            loader = CSVLoader(file_path=data_path)
            docs = loader.load()
            db = FAISS.from_documents(docs, embeddings)
            db.save_local(faiss_path)
            logger.info(f"Index created and saved successfully.")
        else:
            logger.info(f"Loading existing index...")
            db = FAISS.load_local(faiss_path, embeddings)
            logger.info(f"Index loaded successfully.")

        openai.api_key = openai_api_key
        search_results = db.similarity_search(query)

        search_res = search_results[0].page_content
        search_res = search_res[search_res.index("\nsummary: ") + len("\nsummary: "):]
        logger.info(f"Similarity search completed. Result: {search_res}")
        return search_res
    except Exception as e:
        logger.warning(f"An error occurred: {str(e)}")
        logger.warning("Something went wrong during similarity_search")
        return query
