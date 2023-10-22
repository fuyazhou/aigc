import config
from free_dialogue import *
import logging
from flask import Flask, request
from dialogue_service import Dialogue_Service
from flask import Flask, request, jsonify
import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import prompt_template
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 初始化对话服务
dialogue_service = Dialogue_Service()
# dialogue_service.init_source_vector(False)
dialogue_service.init_source_vector(True)
dialogue_service.init_character_dialogue_precision_qa_chain()

llm_summary = ChatOpenAI(openai_api_key=config.openai_api_key)
prompt_summary = PromptTemplate.from_template(prompt_template.template_summary_news)
chain_summary = prompt_summary | llm_summary

app = Flask(__name__)


@app.route('/free_dialogue', methods=['POST'])
def free_dialogue():
    user_id = ""
    try:
        logger.info("*******start free_dialogue server *******")
        data = request.get_json()
        logger.info(f"free_dialogue input  is  {str(data)}")
        user_query = data['query']
        user_id = data["user_id"]
        if check_taiwan(user_query) == 1:
            result = taiwan
        elif check_who(user_query) == 1:
            result = whoami
        else:
            result = get_free_dialogue_answer(user_id, user_query)

        response_data = {
            'id': user_id,  # Generate a unique ID
            'result': result
        }
        logger.info(f"free_dialogue result  is  {str(response_data)}")
        return response_data
    except Exception as e:
        logger.warning(f"An error occurred during free_dialogue: {e}")
        response_data = {
            'id': user_id,  # Generate a unique ID
            'result': common_responses
        }
        return response_data


@app.route('/chat', methods=['POST'])
def chat():
    user_id = ""
    try:
        logger.info("*******start character_dialogue_precision server *******")
        data = request.get_json()
        logger.info(f"character_dialogue_precision input  is  {str(data)}")
        user_query = data['query']
        user_id = data["user_id"]
        if check_taiwan(user_query) == 1:
            result = taiwan
        elif check_who(user_query) == 1:
            result = whoami
        else:
            result = dialogue_service.character_dialogue_precision_qa(user_query)

        response_data = {
            'id': user_id,  # Generate a unique ID
            'result': result
        }
        logger.info(f"character_dialogue_precision result  is  {str(response_data)}")
        return response_data
    except Exception as e:
        logger.warning(f"An error occurred during character_dialogue_precision:{e}")
        response_data = {
            'id': user_id,  # Generate a unique ID
            'result': common_responses
        }
        return response_data


@app.route('/summary', methods=['POST'])
def summary():
    data = request.get_json()

    if 'user_id' not in data or 'content' not in data:
        error_message = "user_id and content must be provided in the input JSON"
        app.logger.error(error_message)
        return jsonify({"error": "user_id and content must be provided in the input JSON"}), 400

    user_id = data['user_id']
    content = data['content']

    res = chain_summary.invoke({"news": content})
    try:
        res1 = json.loads(res.content)
        ch = res1["ch"]
        en = res1["en"]
    except:
        ch = res.content
        en = res.content

    output_data = {"user_id": user_id, "ch": ch, "en": en}
    app.logger.info(f"Processed request for user : {user_id}, res: {res.content}")
    return jsonify(output_data)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5051)
