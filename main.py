from free_dialogue import *
import logging
from flask import Flask, request

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = Flask(__name__)


@app.route('/free_dialogue', methods=['POST'])
def free_dialogue():
    try:
        logger.info("*******start free_dialogue server *******")
        data = request.get_json()
        logger.info(f"free_dialogue input  is  {str(data)}")
        user_query = data['query']
        user_id = data["user_id"]
        if check_taiwan(user_query) == 1:
            result = taiwan
        else:
            result = get_free_dialogue_answer(user_id, user_query)
        response_data = {
            'id': user_id,  # Generate a unique ID
            'result': result
        }
        return response_data
    except Exception as e:
        logger.warning("An error occurred during question answering:")
        return str(e)
