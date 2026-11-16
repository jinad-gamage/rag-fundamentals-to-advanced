# Lesson 01 – LLM Fundamentals

In this lesson, we learn how to use a Large Language Model (LLM) through an API and get a basic introduction to RAG (Retrieval-Augmented Generation).

## What You Will Learn

- What is a Large Language Model (LLM)?
- How to connect to the Gemini API
- How to securely store an API key using environment variables
- How to send a prompt to an LLM and receive a response
- A simple introduction to RAG

## Project Structure

```text
Lesson-01-LLM-Fundamentals/
├── main.py
├── .env                 # Local API key; do not commit this file
├── .gitignore
└── README.md
```

## Setup

Python 3.10 or a later version is required.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install google-genai python-dotenv
```

## Environment Variable

Create a `.env` file and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

Never add your API key directly to the source code or commit it to the Git repository.

## Run the Example

```bash
python main.py
```

The program sends the prompt `What is RAG? Explain simply.` to the Gemini LLM and prints the response in the terminal.
