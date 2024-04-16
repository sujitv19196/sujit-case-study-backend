import os
import json
import tempfile
import time

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

from scripts.gpt4querygen import GPTQueryGen
from whisperplus import SpeechToTextPipeline

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
    stt = SpeechToTextPipeline(model_id="openai/whisper-large-v3")

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
    
    @app.route('/audio', methods=['POST'])
    def audio():
        audio_file = request.files.get('audio_data')
        file_type = request.form.get("type")
        print(audio_file)
        print(file_type)
        filename = "myAudioFile." + file_type
  
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Construct the target path
        target_path = os.path.join(temp_dir, filename)
        
        try:
            # Save the audio file to the temporary directory
            audio_file.save(target_path)
            
            # Run whisper
            start_time = time.time()
            transcript = stt(target_path, "openai/whisper-large-v3", "english")
            whisper_stt_time = time.time() - start_time

            response = gpt.ask(transcript)
            gpt.print_debug_stats()
            print("Whisper STT Time: ", whisper_stt_time)
            
            return json.dumps({"message": response, 
                               "query": transcript})
        
        except Exception as e:
            print(f"Error processing audio: {e}")
            return jsonify({"message": "Error processing audio"}), 500
        
        finally:
            # delete the temporary directory to clean up
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    return app