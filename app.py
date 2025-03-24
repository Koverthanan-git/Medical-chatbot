from flask import Flask, request, jsonify, render_template
import requests
from flask_cors import CORS  # Allow frontend-backend communication

app = Flask(__name__, template_folder="templates")
CORS(app)  # Enable Cross-Origin Resource Sharing

OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama Local API

def generate_response(user_input):
    """Generate medical responses using Mistral LLM via Ollama."""
    medical_prompt = f"You are a medical assistant. Answer concisely:\n{user_input}"

    payload = {
        "model": "mistral",
        "prompt": medical_prompt,
        "stream": False,
        "options": {
            "num_ctx": 160  # Lower context size to reduce RAM usage
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("response", "I'm sorry, I couldn't generate a medical response.")
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handles medical chat requests and returns a bot response."""
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Invalid request"}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    bot_response = generate_response(user_message)
    return jsonify({"reply": bot_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)