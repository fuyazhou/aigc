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
    max_tokens=2048,
)


# from temp_model import *
# chat = chat
# embeddings = embeddings


def generate_data(query,
                  data_path: str = database_path,
                  faiss_path: str = faiss_index_path):
    """
        Generates data for a given query.

        Args:
            query (str): The query.
            data_path (str, optional): Path to the data. Defaults to `database_path`.
            faiss_path (str, optional): Path to the FAISS index. Defaults to `faiss_index_path`.

        Returns:
            None.
    """
    logger.info("Generating data for query: {}".format(query))

    logger.info("Fetching Google search results")
    google_search_results = get_google_search_results(query)
    logger.info("Google search results fetched successfully")

    logger.info("Fetching related questions from Google search results")
    google_related_questions = get_goolge_related_questions(google_search_results)
    logger.info("Related questions fetched successfully")

    logger.info("Fetching organic results from Google search results")
    google_organic_results = get_goolge_organic_results(google_search_results)
    logger.info(f"Organic results fetched successfully =={google_organic_results}")

    logger.info("Saving search content")
    content = save_search_content(query, data_path, google_related_questions, google_organic_results, faiss_path)
    logger.info("Search content saved successfully")
    print(f"Search content saved successfully, result = {content}")
    return content


def generate_article(query,
                     data_path: str = database_path,
                     faiss_path: str = faiss_index_path,
                     article_path: str = article_path):
    """
    Generates an article for a given query.

    Args:
        query (str): The query.
        data_path (str, optional): Path to the data. Defaults to `database_path`.
        faiss_path (str, optional): Path to the FAISS index. Defaults to `faiss_index_path`.

    Returns:
        str: The article.
    """

    # Log the start of the function.
    logger.info("Generating article for query: {}".format(query))

    try:
        print("Generating article for query: {}".format(query))

        # Perform similarity search.
        logger.info("Performing similarity search")
        summary = similarity_search(faiss_path, data_path, query)
        logger.info("Similarity search finished")

        # Generate chat messages.

        # messages = [
        #     SystemMessage(content=generate_prompt),
        #     HumanMessage(content=summary)
        # ]
        # logger.info("Chat messages generated")
        #
        # # Chat with the model.
        # logger.info("Chatting with the model")
        # result = chat(messages)
        # logger.info("Chat finished")
        #
        # # Get the article.
        # logger.info("Getting the article")
        # article = result.content

        logger.info("Generating chat messages")
        article = chat_with_model(generate_prompt, summary)
        print("first Generated article for query: {}".format(article))

        logger.info("Generating chat messages again")
        article = chat_with_model(generate_prompt_2, article)
        print("second  Generated article for query: {}".format(article))

        logger.info("Article fetched successfully")
        # Log the end of the function.
        logger.info("Generated article for query: {}".format(article))

        # 保存结果到数据库
        article_dict = {"query": [query], "summary": [summary], "article": [article]}
        df = pd.DataFrame.from_dict(article_dict)

        if os.path.isfile(article_path):
            logger.info("Merging content with existing article")
            df1 = pd.read_csv(article_path)
            df = pd.concat([df1, df])
        logger.info("Saving df to article path")
        df.to_csv(article_path, index=False, encoding="utf_8")

        return article
    except Exception as e:
        logger.warning("Generating article something wrong")
        logger.warning(f"An error occurred: {str(e)}")
        return "Generating article something wrong"


def chat_with_model(generate_prompt, summary):
    logger = logging.getLogger(__name__)
    messages = [
        SystemMessage(content=generate_prompt),
        HumanMessage(content=summary)
    ]
    logger.info("Chat messages generated")

    try:
        # Chat with the model.
        logger.info("Chatting with the model")
        result = chat(messages)
        article = result.content
        return article

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return summary


def main(generate_data_query: str = "None", generate_article_query: str = "None"):
    if generate_data_query != "None":
        print("\n==================================\n")
        print(f"start generate data query is {generate_data_query}")
        data = generate_data(generate_data_query)
        print(f"generate data done, data =  {data}")
    if generate_article_query != "None":
        print("\n==================================\n")
        print(f"start article data query is {generate_article_query}")
        data = generate_article(generate_article_query)
        print(f"generate article done, data =\n\n  {data}")
        print("\n\n")

    if generate_data_query == "None" and generate_article_query == "None":
        print("\n==================================\n")
        print("please input generate_data_query or generate_article_query")


# main.py --generate_data_query 知乎的商業模式 --generate_article_query 知乎的商業模式
if __name__ == '__main__':
    fire.Fire(main)
    # generate_data('知乎的商業模式')
    # article = generate_article('知乎的商業模式')
