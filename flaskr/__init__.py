import os
import json

from flask import Flask
from flask_cors import CORS, cross_origin

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    cors = CORS(app)
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

    # a simple page that says hello
    @app.route('/hello', methods=['GET'])
    def hello():
        print("Hello, World on BACKEND!")
        return json.dumps({"message": "Hello, World!"})

    # @app.route('/gpt4')
    # def gpt4():
    #     embeddings_path = "https://cdn.openai.com/API/examples/data/winter_olympics_2022.csv"
    #     df = pd.read_csv(embeddings_path)
    #     df['embedding'] = df['embedding'].apply(ast.literal_eval)


    
    return app