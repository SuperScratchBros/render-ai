# Render AI

A small AI web app (FastAPI + a single `index.html`) that runs on Render. It offers several chat providers, optional web search, image generation, private file uploads, a community chat, and a live usage/limits dashboard.

## Features

| Feature | Provider(s) |
| --- | --- |
| Chat | Groq, Google Gemini, OpenRouter, xKiro free models, NVIDIA NIM free chat endpoints, and KiosAPI chat models |
| Fast Search | Tavily |
| Deep Search | Exa |
| Write Code | Groq in chat mode; dedicated NVIDIA Code Studio with curated free code-capable models |
| Image generation | NVIDIA FLUX.2 Klein 4B, Cloudflare FLUX, or xKiro free image models |
| File uploads | Supabase Storage |
| Usage, limits and community chat | Supabase |

### KiosAPI

[KiosAPI](https://kiosapi.id/docs) is an OpenAI-compatible API gateway. With `KIOSAPI_API_KEY` set in Render, the app fetches the live `/v1/models` catalog and lists its chat-capable text models in the main model picker. Model IDs stay namespaced as `kiosapi:<provider/model>` internally, and the API key is only used by the server. Embedding, ranking, image, video, audio, and other non-chat endpoints are excluded from the chat picker. KiosAPI pricing and account access still apply to each selected model.

### xKiro

[xKiro](https://xkiro.com) is an OpenAI-compatible gateway with many models behind one API key.

- **Free models only.** The app loads xKiro's public model list (`GET /v1/models`) and keeps only models whose `access_tier` is `free`. Paid/premium models are not listed and are rejected by the server even if someone sends the ID by hand.
- **Pick the exact model.** Choose "xKiro: free models" and use the dropdown. Leaving it on Default uses the first free model (or `XKIRO_MODEL` if set). The reply shows which model actually answered.
- **Image engine toggle.** In Image Studio, choose NVIDIA FLUX.2 Klein 4B, Cloudflare FLUX, or xKiro. xKiro image jobs are asynchronous, so the page shows progress and keeps checking until the image is ready (up to 5 minutes).
- **NVIDIA chat models.** The chat model list includes NVIDIA's chat-capable free endpoints, including models for reasoning, vision, and general chat.
- **Code Studio.** A separate sidebar panel exposes NVIDIA's free code-capable models, including GLM-5.3, DeepSeek V4.1 Flash, Kimi K3, Gemma 4 31B IT, and Poolside Laguna XS 2.1. Availability and free-trial quotas are controlled by NVIDIA and can change.
- **Fallback.** If xKiro is rate-limited or out of free allowance, chat falls back to OpenRouter or Groq when those keys are set.

## Environment variables

Set these in the Render dashboard (**Service → Environment**). Secrets use `sync: false` in `render.yaml`, so Render will ask for them but they are never stored in the repo. **Never commit API keys.**

### Required

| Variable | What it is |
| --- | --- |
| `APP_SECRET_KEY` | Long random string used to sign the anonymous user cookie. |
| `SUPABASE_URL` | Your Supabase project URL. |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service-role key (server only). |
| At least one chat key | `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `XKIRO_API_KEY` or `KIOSAPI_API_KEY`. |

### Provider keys

| Variable | Enables |
| --- | --- |
| `GROQ_API_KEY` | Groq chat and Write Code mode. |
| `GEMINI_API_KEY` | Google Gemini chat. |
| `OPENROUTER_API_KEY` | OpenRouter chat (also used as a fallback). |\n| `KIOSAPI_API_KEY` | KiosAPI live chat-model catalog and model completions. Set this key privately in Render; never commit it. |
| `XKIRO_API_KEY` | xKiro free chat models and xKiro image generation. |
| `NVIDIA_API_KEY` | NVIDIA NIM chat models, NVIDIA Code Studio, and FLUX.2 Klein 4B image generation. Create it at [build.nvidia.com](https://build.nvidia.com/). Add the key privately in Render → Environment; never commit it. |
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
| `XKIRO_MODEL` | first free model | Default xKiro chat model. Must be a free model. |\n| `KIOSAPI_BASE_URL` | `https://api.kiosapi.id/v1` | KiosAPI OpenAI-compatible API base URL. |\n| `KIOSAPI_MODEL` | `deepseek/deepseek-v4-flash` | Fallback/default KiosAPI chat model if its live catalog cannot be reached. |
| `XKIRO_IMAGE_MODEL` | first free image model (`sensenova/sensenova-u1.5-lite`) | xKiro image model. Must be a free model. |
| `XKIRO_BASE_URL` | `https://api.xkiro.com/v1` | xKiro API base URL. |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA's OpenAI-compatible chat API. |
| `NVIDIA_IMAGE_MODEL` | `black-forest-labs/flux.2-klein-4b` | NVIDIA hosted image-generation model. |
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
| `XKIRO` | xKiro free chat | 200 | 6000 | 15 | 3 |\n| `KIOSAPI` | KiosAPI chat | 200 | 6000 | 15 | 3 |
| `XKIRO_IMAGE` | xKiro free images | 60 | 1800 | 3 | 1 |
| `NVIDIA` | NVIDIA NIM chat and Code Studio | 300 | 9000 | 12 | 3 |
| `NVIDIA_IMAGE` | NVIDIA FLUX.2 Klein 4B images | 45 | 1350 | 3 | 1 |
| `TAVILY` | Fast Search | 100 | 1000 | 5 | 2 |
| `EXA` | Deep Search | 20 | 600 | 3 | 1 |
| `CLOUDFLARE` | Cloudflare images | 90 | 2700 | 3 | 1 |

The usage dashboard in the app shows daily and monthly remaining requests for each provider, with explicit reset timing. Usage is calculated from the current calendar period, so monthly counters roll over automatically on the first day of each month.

## Saved AI chats

Each anonymous user can have up to **3 saved AI chats**. Each chat keeps only its **5 most recent messages**. The server loads history by chat ID, so messages from the other chats are never fed to the AI. For an existing Supabase project, run `supabase_ai_chats.sql` once; for a fresh setup, `supabase_schema.sql` includes these tables.

## Supabase

The app stores usage, users, saved AI chat history, community chat and file metadata in Supabase (`render_users`, `render_usage`, `render_files`, `render_ai_chats`, `render_ai_chat_messages`, `chat_users`, `chat_sessions`, `chat_messages`). Run `supabase_schema.sql` once in the Supabase SQL editor for a fresh setup. `supabase_ai_chats.sql` is the migration for an existing setup.

## Run locally

```bash
pip install -r requirements.txt
export APP_SECRET_KEY=change-me
export SUPABASE_URL=...
export SUPABASE_SERVICE_ROLE_KEY=...
export XKIRO_API_KEY=...   # plus any other provider keys you want
uvicorn main:app --reload
```
