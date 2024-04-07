import os
import json

from flask import Flask, request
from flask_cors import CORS, cross_origin

from .gpt4querygen import GPTQueryGen
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app)
    app.config.from_mapping(
        SECRET_KEY='dev',
        # DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
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

    # init vector db 
    # db = load_faiss("../faiss/faiss_index_hf_v6") # TODO make arg? 
   
    # init gpt model 
    gpt = GPTQueryGen(model="gpt-3.5-turbo", embeddings = "hf", db_name="./faiss_index", token_budget=4096)

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
            response = gpt.ask(query, print_message=False)
            print(response)
            return json.dumps({"message": response})
            
    return app

def load_faiss(db_name):
    embeddings = None
    embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'mps'}
    encode_kwargs = {'batch_size': 8}
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    db = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)
    return db 