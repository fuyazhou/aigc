from free_dialogue import *
import logging
from flask import Flask, request
from dialogue_service import Dialogue_Service

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 初始化对话服务
dialogue_service = Dialogue_Service()
dialogue_service.init_source_vector()
dialogue_service.init_character_dialogue_precision_qa_chain()

# query = "你好"
# result = dialogue_service.character_dialogue_precision_qa(query)
# print(result)
# query = "人才补贴"
# result = dialogue_service.character_dialogue_precision_qa(query)
# print(result)
# query = "残疾人"
# result = dialogue_service.character_dialogue_precision_qa(query)
# print(result)

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


@app.route('/character_dialogue_precision', methods=['POST'])
def character_dialogue_precision():
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


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
