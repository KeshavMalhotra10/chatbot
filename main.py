# import the OpenAi libraries
import openai
from openai import OpenAI

# allows us to load environment variable (.env) for security
import os
from dotenv import load_dotenv

# for vector math, we use it in dot product similarity for RAG retrieval
import numpy as np

# flask is a backend web framework
# request reads the incoming JSON
# jsonify sends the json back
# render_template delivers the hmtl
# cors frontend can talk to the backend
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# create a flask instance
app = Flask(__name__)
CORS(app)

# securely loading key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# setting up openAI key
client = OpenAI()

# here I load knowledge for keshavBot based on some personal info
with open("info.txt", "r") as file:
    keshav_info = file.read()

# RAG SECTION


# splits up big text into chunks of 500 characters
def chunk_test(text, chunk_size=500):
    # text[i: i+chunk_size] takes slices of the text and we move i from the range of 0 to 1200 with increments of 500
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


# splits my info file into smaller chunks
chunks = chunk_test(keshav_info)

# create the embeddings for each chunk
chunk_embeddings = []
for chunk in chunks:
    # get a vector representation
    emb = client.embeddings.create(model="text-embedding-3-small", input=chunk)
    # store this vector representation in a vector list
    chunk_embeddings.append(emb.data[0].embedding)

# converts list of vectors into a usable NumPy array
chunk_embeddings = np.array(chunk_embeddings)


# retrieval function which takes the question and returns 3 relevant chunks
def retrieve_relevant_chunks(query, k=3):
    # user question turned into embedding vector
    query_embedding = (
        client.embeddings.create(model="text-embedding-3-small", input=query)
        .data[0]
        .embedding
    )

    # embedding vecgtor turned into NumPy array
    query_vector = np.array(query_embedding)

    # Cosine similarity
    # not perfect, but if dot product has higher value, it has higher similarity
    similarities = np.dot(chunk_embeddings, query_vector)

    # sorts the similarities, and selects the top k most similar
    top_k_indices = similarities.argsort()[-k:][::-1]

    # return the relevant chunks
    return [chunks[i] for i in top_k_indices]


# System prompt
system_prompt = f"""
You are KeshavBot.

Follow these rules:
- Use only the provided context to answer.
- If the context does not contain the answer, say you don't know.
- Speak casually and bluntly like Keshav.
- Maximum 3 sentences.
- Under 100 tokens.
- Add a reputable "Learn more: <url>" link for technical questions.
"""

# --- Flask Route ---


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():  # This function is only used when chatting
    data = request.get_json()
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    relevant_chunks = retrieve_relevant_chunks(user_input)
    context = "\n\n".join(relevant_chunks)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{user_input}",
        },
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-3.5-turbo"
            messages=messages,
            max_tokens=100,
            temperature=0.7,
        )
        bot_response = response.choices[0].message.content.strip()
        return jsonify({"reply": bot_response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Something went wrong"}), 500


# --- Run Flask ---
if __name__ == "__main__":
    app.run(debug=True, port=5001)
