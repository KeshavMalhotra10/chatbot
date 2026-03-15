# JobFit — AI Resume Analyzer

JobFit is a RAG-powered web app that analyzes your resume against any job posting and returns an ATS match score, missing keywords, and actionable feedback. Built with Flask, OpenAI, and NumPy.

**[Live Demo →](https://jobfit-3tgb.onrender.com/)**

---

## How It Works

1. You paste a job posting into the app
2. The backend uses **Retrieval-Augmented Generation (RAG)** to find the most semantically relevant sections of your resume using vector embeddings and cosine similarity
3. Those chunks are injected into a prompt alongside the job posting
4. GPT returns a structured JSON response with a match score, strong matches, missing ATS keywords, and a tip
5. The frontend renders the results with a match dial, keyword importance badges, and the retrieved resume chunks with similarity scores

---

## Features

- **Match Score** — 0–100 ATS compatibility score with reasoning
- **Missing Keywords** — keywords from the job posting absent from your resume, ranked by importance
- **Strong Matches** — resume experience that directly aligns with the role
- **RAG Transparency** — see exactly which resume chunks were retrieved and their cosine similarity scores
- **Input Validation** — rejects vague or short inputs to prevent hallucinated results

---

## Tech Stack

- **Backend:** Python, Flask, Flask-CORS
- **AI:** OpenAI GPT-3.5-turbo, text-embedding-3-small
- **RAG:** NumPy cosine similarity, sentence-boundary chunking
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Render

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/KeshavMalhotra10/chatbot.git
cd chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
Create a `.env` file in the root:
```
OPENAI_API_KEY=your_openai_key_here
```

### 5. Change RESUME in app.py
```bash
RESUME = """ ... """
```

### 6. Run the app
```bash
python app.py
```

Then open `http://localhost:5001` in your browser.

---

## Project Structure

```
chatbot/
├── app.py              # Flask backend — RAG pipeline + /analyze endpoint
├── templates/
│   └── index.html      # Frontend UI
├── requirements.txt    # Dependencies
├── Procfile            # Render deployment config
├── .env                # API key (gitignored)
└── .gitignore
```

---

## RAG Implementation

The retrieval pipeline works as follows:

- Resume text is split into sentence-boundary-aware chunks (~400 chars)
- Each chunk is embedded using OpenAI's `text-embedding-3-small` model at startup
- On each request, the job posting is embedded and compared against all chunk embeddings via dot product (cosine similarity)
- The top-4 most relevant chunks are injected into the prompt as context

This grounds the LLM's analysis in actual resume content rather than generating generic feedback.
