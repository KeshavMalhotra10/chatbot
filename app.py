# importing all necessary libraries
import openai
import re
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

RESUME = """
Keshav Malhotra
226-505-9512 | keshav.malhotra@uwaterloo.ca | linkedin.com/in/keshavxmalhotra | github.com/KeshavMalhotra10
keshavmalhotra.com

EDUCATION
University of Waterloo, 2025–Present
BASc Mechatronics Engineering (Co-op) — Waterloo, ON
George A. Ward Entrance Scholarship | President's Scholarship of Distinction

SKILLS
Languages: C++, Python, Java, JavaScript, HTML/CSS, SQL
Full Stack: React, Node.js, WordPress
Tools: Git, PyTorch, VS Code, PyCharm, IntelliJ, CLion, Eclipse, pandas, NumPy, Matplotlib, Langchain
Other: Linux, MS Office, G Suite, Azure

EXPERIENCE
SickKids Hospital — AI Research Intern (Jul 2024 – Oct 2024), Toronto ON
- Contributed to development and launch of SKAI (SickKids AI), the hospital's first AI chatbot
- Influenced LLM optimization for clinical deployment by synthesizing qualitative data from 100+ patient interviews
- Prototyped a mobile autonomous robot for emergency department logistics with sensor fusion and Arduino-based control

FRC Team 6141 — School Captain (Jun 2024 – Jun 2025), Toronto ON
- Developed Java-based control system for differential drive robot using WPILib
- Mentored 30+ students in Git, OOP, sensor integration

Ideal Computers Technology — Technical Team Intern (Nov 2023 – Dec 2023)
- Website redesign using HTML, CSS, JavaScript
- OS installation/config (Windows, MacOS, Linux)
- Diagnosed and fixed system crashes via CLI

PROJECTS
NavBot — Autonomous Hospital Navigation Robot (C++, Git) Nov 2025 – Dec 2025
- Autonomous navigation software, real-time decision-making in dynamic indoor environments
- Dijkstra's algorithm for global path planning + PI controller for differential drive motors
- Rotational accuracy within 1 degree error, linear precision within 5cm

Roboventure (Arduino, Git, Excel) Aug 2024 – Sept 2024
- Technical workshops for 50+ students in Arduino firmware, circuit prototyping
- Scaled org participation 300% in one month; secured $1,500 from Promise1000 Canada

AI Heart Disease Prediction (Python, NumPy, Matplotlib, Kaggle) Oct 2023 – Nov 2023
- Logistic Regression classifier from scratch using NumPy; 92% accuracy on heart disease dataset
- Vectorized training logic for real-time cost function visualization
"""


def chunk_text(text, size=400):
    current = ""
    sentences = []
    text = text.split(".")
    for c in text:
        if len(current) < size:
            current += c
        elif len(current) >= size:
            sentences.append(current)
            current = c
    if current:
        sentences.append(current)
    return sentences


chunks = chunk_text(RESUME)
chunk_embeddings = []
for c in chunks:
    embed = client.embeddings.create(model="text-embedding-3-small", input=c)
    chunk_embeddings.append(embed.data[0].embedding)
chunk_embeddings = np.array(chunk_embeddings)


def retrieve_relevant_chunks(query, k=4):
    queryEmbed = client.embeddings.create(model="text-embedding-3-small", input=query)
    queryVector = np.array(queryEmbed.data[0].embedding)
    similarity = np.dot(chunk_embeddings, queryVector)
    top = similarity.argsort()[-k:][::-1]
    return [{"chunk": chunks[i], "score": round(float(similarity[i]), 4)} for i in top]


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    job_posting = data.get("job_posting", "").strip()

    if not job_posting:
        return jsonify({"error": "No job posting provided"}), 400

    if len(job_posting) < 100:
        return (
            jsonify({"error": "Job posting is too short — paste the full description"}),
            400,
        )

    retrieved = retrieve_relevant_chunks(job_posting)
    context = "\n\n".join([r["chunk"] for r in retrieved])

    prompt = f"""
You are an expert ATS (Applicant Tracking System) analyst and career coach.

Here is the candidate's resume (relevant sections):
{context}

Here is the job posting:
{job_posting}

Analyze and return a JSON object with EXACTLY this structure (no markdown, no extra text):
{{
  "match_score": <integer 0-100>,
  "match_reasoning": "<2 sentences explaining the score>",
  "strong_matches": ["<skill or experience that directly matches>", ...],
  "missing_keywords": [
    {{"keyword": "<keyword>", "importance": "high|medium|low", "context": "<why it matters for this role>"}},
    ...
  ],
  "ats_tip": "<one concrete sentence on the single most impactful change to make>"
}}

Rules:
- missing_keywords should be 5-8 keywords/phrases from the job posting NOT present in the resume
- strong_matches should be 4-6 genuine matches
- Be specific, not generic. Reference actual resume content.
- Base your analysis ONLY on what is explicitly written in the job posting. Do not invent requirements.
- If the job posting is vague or too short, return a match_score of 0 and explain why in match_reasoning.
- Treat development skills as transferable. A candidate with frontend (React, JS, HTML/CSS) and backend (Node.js, Python, SQL) experience should be considered capable of mobile and web development, even if not explicitly listed.
- Return only valid JSON. No extra text before or after.
- No trailing commas anywhere in the JSON — trailing commas will break the parser and are strictly forbidden.
- The JSON must be parseable by Python's json.loads() with zero modifications.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        # Remove trailing commas before ] or } (GPT does this sometimes)
        raw = re.sub(r",\s*([}\]])", r"\1", raw)

        result = json.loads(raw)
        result["sources"] = retrieved
        return jsonify(result)
    except json.JSONDecodeError as e:
        print("JSON ERROR:", e)
        return jsonify({"error": f"Failed to parse response: {str(e)}"}), 500
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host="0.0.0.0", port=port)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host="0.0.0.0", port=port)
