import openai
from openai import OpenAI
import os 
from dotenv import load_dotenv

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- Setup ---
app = Flask(__name__)
CORS(app)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Load Keshav's background info
with open("info.txt", "r") as file:
    keshav_info = file.read()

# System prompt
system_prompt = f"""
You are KeshavBot, a helpful and friendly AI assistant who knows everything about Keshav.
Keshav is your creator. Here is his background:

{keshav_info}
You speak casually, use phrases like "yo!" sometimes, and provide blunt but personalized responses. You make sure your responses are only one sentence long and are less than 20 tokens.
"""

# --- Flask Route ---

@app.route("/")
def index():
    return render_template("index.html")




@app.route("/chat", methods=["POST"])
def chat(): #This function is only used when chatting 
    data = request.get_json()
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo"
            messages=messages,
            max_tokens=25,
            temperature=0.7
        )
        bot_response = response.choices[0].message.content.strip()
        return jsonify({"reply": bot_response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Something went wrong"}), 500

# --- Run Flask ---
if __name__ == "__main__":
    app.run(debug=True)
