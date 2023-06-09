from flask import Flask, request, jsonify
from main import *

app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate():
    generate_data_query = request.json.get('generate_data_query', 'None')
    generate_article_query = request.json.get('generate_article_query', 'None')
    result = {}

    if generate_data_query != "None":
        result['generate_data'] = generate_data(generate_data_query)

    if generate_article_query != "None":
        result['generate_article'] = generate_article(generate_article_query)

    if generate_data_query == "None" and generate_article_query == "None":
        return "Please input generate_data_query or generate_article_query"

    return jsonify(result)


if __name__ == '__main__':
    app.run()
