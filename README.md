# Render AI

A minimal AI web app with three model choices,Chat Gpt 4.0,Gemini-3.8-flash,and Openrouter. It also has an optional Tavily web search. It is designed to run on Render Free trial.

## Models

- Groq — `openai GPT 4.0`
- Gemini — `gemini-3.8-flash`
- OpenRouter — `openrouter/free`

## Required environment variables

`GROQ_API_KEY`
`GEMINI_API_KEY`
`OPENROUTER_API_KEY`
`TAVILY_API_KEY`

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
