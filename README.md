# NLGEP AI

Small FastAPI app for Render's free tier. Pick a model, then turn on tools.

**Models**
- OpenAI: GPT 4.0 (runs on Groq, key `GROQ_API_KEY`)
- Gemini: 3.8 Flash (`GEMINI_API_KEY`)
- OpenRouter: Mixed (`OPENROUTER_API_KEY`)

**Tools (toggle buttons)**
- Fast Search: Tavily (`TAVILY_API_KEY`)
- Deep Search: Exa (`EXA_API_KEY`), slower and uses more credits
- Write Code: always answers with Groq, using a code-focused prompt

Fast and Deep Search can be on together. Everything is optional: a model or tool without its key shows as "not set up".

**Optional settings:** `GROQ_MODEL`, `GROQ_CODE_MODEL`, `GEMINI_MODEL`, `OPENROUTER_MODEL`, `OPENROUTER_SITE_URL`, `OPENROUTER_APP_NAME`.

Set keys in Render under Environment, never in the repo.
