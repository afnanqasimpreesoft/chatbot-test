from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

# Define the repository and filename for the model
repo_id = "Nistep/bitnet_b1_58-3B-Q4_K_M-GGUF"
filename = "bitnet_b1_58-3b-q4_k_m.gguf"

# Download the model file
model_path = hf_hub_download(repo_id=repo_id, filename=filename)
llm = Llama(model_path=model_path)

# Initialize Flask app
app = Flask(__name__, static_folder="static")
CORS(app)  # Enable CORS for all routes

# Route to serve the frontend HTML page
@app.route("/")
def serve_html():
    return send_from_directory("static", "chatbot.html")

# Define the API endpoint to generate responses
@app.route("/generate", methods=["POST"])
def generate_response():
    try:
        data = request.get_json()
        prompt = data["prompt"]
        
        # Generate response from the model
        response = llm(prompt)
        
        # Return the generated text
        return jsonify({"response": response["choices"][0]["text"]})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Root endpoint to return a greeting (optional)
@app.route("/hello", methods=["POST"])
def generate_hello():
    return jsonify({"message": "hi how are you"})

if __name__ == "__main__":
    # Start the Flask app on port 8000
    app.run(host="0.0.0.0", port=8000)
