import os
import json
import signal

from flask import Flask, request
from flask_cors import CORS, cross_origin

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from scripts.gpt4querygen import GPTQueryGen

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )
    app.config["CORS_HEADERS"] = "Content-Type"

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # init gpt model 
    gpt = GPTQueryGen(model="gpt-3.5-turbo", embeddings = "hf", db_name="faiss/", token_budget=8192, debug=True)

    # a simple page that says hello!
    @app.route('/hello', methods=['GET'])
    def hello():
        print("Hello, World on BACKEND!")
        return json.dumps({"message": "Hello, World!"})

    @app.route('/ask', methods=['POST'])
    def ask():
        data = request.json  # Parse JSON data
        query = data.get('query')
        if not query is None:
            response = gpt.ask(query)
            gpt.print_debug_stats()
            return json.dumps({"message": response})
    
    return app