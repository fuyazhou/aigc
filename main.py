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

def append_to_file(file_path, new_content):
    with open(file_path, 'a', encoding="utf_8") as f:
        f.write(new_content)


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
    print("\n\n==================================\n\n")
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
        print("\n\n==================================\n\n")
        print("Generating article for query: {}".format(query))
        print("\n\n==================================\n\n")

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
        print("\n\n==================================\n\n")
        print("first Generated article for query:\n\n {}".format(article))
        print("\n\n==================================\n\n")

        logger.info("****Generating chat messages again***")
        article = chat_with_model(generate_prompt_2, article)

        logger.info("****translate  article to Chinese* **")
        cn_article = chat_with_model("You are a helpful assistant that translates English to Chinese.", article)

        print("\n\n==================================\n\n")
        print("second  Generated article for query:\n\n {}".format(article))
        print("\n\n==================================\n\n")
        print("\n\n==================================\n\n")
        print("translate second  Generated article into chinese:\n {}".format(cn_article))
        print("\n\n==================================\n\n")

        logger.info("Article fetched successfully")
        # Log the end of the function.
        logger.info("Generated article for query: {}".format(article))
        logger.info("translate second  Generated article into chinese:\n {}".format(cn_article))

        # 保存结果到数据库
        article_dict = {"query": [query], "summary": [summary], "article": [article + "\n\n\n\n" + cn_article]}
        df = pd.DataFrame.from_dict(article_dict)

        if os.path.isfile(article_path):
            logger.info("Merging content with existing article")
            df1 = pd.read_csv(article_path)
            df = pd.concat([df1, df])
        logger.info("Saving df to article path")
        df.to_csv(article_path, index=False, encoding="utf_8")

        logger.info("save article  to  data/article.txt")
        append_to_file(article_txt_path, "****************\n")
        append_to_file(article_txt_path, query + ":\n\n")
        append_to_file(article_txt_path, article + "\n\n")
        append_to_file(article_txt_path, cn_article + "\n\n")

        return article + "\n\n\n\n" + cn_article
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
        # data = "\n".join(data)
        # print(f"generate data done, data =  {data}")
    if generate_article_query != "None":
        # print("\n==================================\n")
        # print(f"start article data query is {generate_article_query}")
        data = generate_article(generate_article_query)
        # print("\n\n==================================\n\n")
        # print(f"generate article done, data =\n\n  {data}")
        # print("\n\n")

    if generate_data_query == "None" and generate_article_query == "None":
        print("\n==================================\n")
        print("please input generate_data_query or generate_article_query")


# main.py --generate_data_query Apple company business model --generate_article_query Apple company business model
if __name__ == '__main__':
    fire.Fire(main)
    # generate_data('Apple company business model')
    # article = generate_article('Apple company business model')

# python main.py --generate_data_query Apple company business model
