# NLGEP AI Minimal

A deliberately minimal AI web app with exactly three model choices and optional Tavily web search.

## Models

- Groq — `openai/gpt-oss-120b`
- Gemini — `gemini-3.8-flash`
- OpenRouter — `openrouter/free`

## Required environment variables

`GROQ_API_KEY`
`GEMINI_API_KEY`
`OPENROUTER_API_KEY`
`TAVILY_API_KEY`

No Supabase, social chat, accounts, sessions, attachments, voice, image generation, or local chat history is included.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
