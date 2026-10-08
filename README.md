# Render AI

A small AI web app (FastAPI + a single `index.html`) that runs on Render. It offers several chat providers, optional web search, image generation, private file uploads, a community chat, and a live usage/limits dashboard.

## Features

| Feature | Provider(s) |
| --- | --- |
| Chat | Groq (`openai/gpt-oss-120b`), Google Gemini, OpenRouter (`openrouter/free`), **xKiro (free models only)** |
| Fast Search | Tavily |
| Deep Search | Exa |
| Write Code | Groq |
| Image generation | **Two options:** Cloudflare Workers AI (FLUX) or xKiro (free image model) |
| File uploads | Supabase Storage |
| Usage, limits and community chat | Supabase |

### xKiro

[xKiro](https://xkiro.com) is an OpenAI-compatible gateway with many models behind one API key.

- **Free models only.** The app loads xKiro's public model list (`GET /v1/models`) and keeps only models whose `access_tier` is `free`. Paid/premium models are not listed and are rejected by the server even if someone sends the ID by hand.
- **Pick the exact model.** Choose "xKiro: free models" and use the dropdown. Leaving it on Default uses the first free model (or `XKIRO_MODEL` if set). The reply shows which model actually answered.
- **Image engine toggle.** In Generate Image mode, switch between Cloudflare FLUX and xKiro. xKiro image jobs are asynchronous, so the page shows progress and keeps checking until the image is ready (up to 5 minutes).
- **Fallback.** If xKiro is rate-limited or out of free allowance, chat falls back to OpenRouter or Groq when those keys are set.

## Environment variables

Set these in the Render dashboard (**Service → Environment**). Secrets use `sync: false` in `render.yaml`, so Render will ask for them but they are never stored in the repo. **Never commit API keys.**

### Required

| Variable | What it is |
| --- | --- |
| `APP_SECRET_KEY` | Long random string used to sign the anonymous user cookie. |
| `SUPABASE_URL` | Your Supabase project URL. |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service-role key (server only). |
| At least one chat key | `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY` or `XKIRO_API_KEY`. |

### Provider keys

| Variable | Enables |
| --- | --- |
| `GROQ_API_KEY` | Groq chat and Write Code mode. |
| `GEMINI_API_KEY` | Google Gemini chat. |
| `OPENROUTER_API_KEY` | OpenRouter chat (also used as a fallback). |
| `XKIRO_API_KEY` | xKiro free chat models and xKiro image generation. |
| `TAVILY_API_KEY` | Fast Search. |
| `EXA_API_KEY` | Deep Search. |
| `CLOUDFLARE_API_TOKEN` | Cloudflare image generation (needs Workers AI access). |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account for image generation. |
| `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` | File metadata and private file storage. |

### Model and behavior settings (all optional)

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model ID. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model ID. |
| `OPENROUTER_MODEL` | `openrouter/free` | OpenRouter model ID. |
| `XKIRO_MODEL` | first free model | Default xKiro chat model. Must be a free model. |
| `XKIRO_IMAGE_MODEL` | first free image model (`sensenova/sensenova-u1.5-lite`) | xKiro image model. Must be a free model. |
| `XKIRO_BASE_URL` | `https://api.xkiro.com/v1` | xKiro API base URL. |
| `CLOUDFLARE_IMAGE_MODEL` | `@cf/black-forest-labs/flux-2-klein-4b` | Cloudflare image model. |
| `IMAGE_DEFAULT_PROVIDER` | `cloudflare` | Image engine selected by default: `cloudflare` or `xkiro`. If that engine isn't configured, the other one is used. |
| `AI_TIMEZONE` | `America/New_York` | Time zone given to the AI for the current date/time. |
| `OPENROUTER_SITE_URL` | empty | Sent to OpenRouter as the referer. |
| `OPENROUTER_APP_NAME` | `Render AI` | Sent to OpenRouter as the app title. |
| `ENVIRONMENT` | empty | Set to `production` or `render` to mark cookies secure. |
| `SECURE_COOKIES` | `false` | Set to `true` to force secure cookies. |
| `BLOB_MAX_FILE_SIZE` | `26214400` (25 MB) | Max upload size in bytes. |
| `PYTHON_VERSION` | `3.12.3` | Python version used by Render. |

### Usage limits (all optional)

Every provider has four limits. The variable name is the provider prefix plus the suffix:

- `<PREFIX>_DAILY_LIMIT` and `<PREFIX>_MONTHLY_LIMIT`: app-wide request caps shared by everyone.
- `<PREFIX>_USER_DAILY_LIMIT` and `<PREFIX>_USER_PER_MINUTE_LIMIT`: per-visitor caps.

| Prefix | Used for | Daily | Monthly | Per user / day | Per user / min |
| --- | --- | --- | --- | --- | --- |
| `GROQ` | Groq chat and code | 900 | 27000 | 10 | 3 |
| `GEMINI` | Gemini chat | 18 | 540 | 4 | 2 |
| `OPENROUTER` | OpenRouter chat | 45 | 1350 | 10 | 3 |
| `XKIRO` | xKiro free chat | 200 | 6000 | 15 | 3 |
| `XKIRO_IMAGE` | xKiro free images | 60 | 1800 | 3 | 1 |
| `TAVILY` | Fast Search | 100 | 1000 | 5 | 2 |
| `EXA` | Deep Search | 20 | 600 | 3 | 1 |
| `CLOUDFLARE` | Cloudflare images | 90 | 2700 | 3 | 1 |

The usage dashboard in the app shows each provider's remaining requests with progress bars (your limits and everyone's), plus the xKiro account's free-token allowance for the day.

## Supabase

The app stores usage, users, community chat and file metadata in Supabase (`render_users`, `render_usage`, `render_files`, `chat_users`, `chat_sessions`, `chat_messages`). `supabase_render_files.sql` creates the files table. If your `render_usage` table restricts which `provider` values are allowed, add `xkiro` and `xkiro_image`.

## Run locally

```bash
pip install -r requirements.txt
export APP_SECRET_KEY=change-me
export SUPABASE_URL=...
export SUPABASE_SERVICE_ROLE_KEY=...
export XKIRO_API_KEY=...   # plus any other provider keys you want
uvicorn main:app --reload
```
