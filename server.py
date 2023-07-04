from flask import Flask, request, jsonify
from main import *

app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate_article():
    generate_data_query = request.json.get('generate_data_query')

    if generate_data_query:
        summary = generate_data(generate_data_query)
        article = generate_article(summary)
        return {'article': article}

    return {'error': 'Please provide a generate_data_query parameter.'}


if __name__ == '__main__':
    app.run()