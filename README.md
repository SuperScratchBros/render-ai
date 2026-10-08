# Render AI

A small AI web app (FastAPI + a single `index.html`, no build step) for Render. It has a standard AI layout: a left sidebar and the chat taking the middle and right.

## Pages (sidebar)

| Page | What it does |
| --- | --- |
| **Chat** | Chat with the selected model. Modes: Chat, Fast Search, Deep Search, Write Code, Deep Think. Copy replies, stop a reply mid-way, attach a text/code file. |
| **AI Models** | Models grouped by provider (Groq, Google Gemini, OpenRouter, xKiro). Click a model to use it. xKiro lists every **free** model with search. |
| **API Limits** | Donut charts showing how much is left per provider. Switch between *You* / *Everyone* and *Today* / *This month*. Also shows the xKiro account's free-token allowance. |
| **Image Generator** | Prompt box with a switchable engine: Cloudflare FLUX or xKiro (free image model). |
| **Image Models** | Pick the image engine and model. |
| **Files** | Drag-and-drop uploads, download, delete, storage meter, and "Use in chat" for text/code files. |
| **Themes & Settings** | Styles (Classic, Glassmorphism, Neumorphism, Flat Minimal), light/dark, theme color, custom instructions, export/clear memory. |
| **Community Chat** | Simple shared chat room (Supabase). |

## Memory

- The chat shows only the **5 most recent exchanges** (your message + the AI reply), and exactly those are sent to the AI as memory.
- Memory is **wiped every Monday** (local time of the visitor's browser). The sidebar shows how long until the next wipe.
- Memory is stored in the visitor's browser (`localStorage`), not on the server. The server also only reads the last 10 history messages.
- Extras: *New chat* button, export the chat as Markdown, and custom instructions (500 characters) that are sent with every message.

## Files

Files are stored in a **private Supabase Storage bucket** (default `render-files`, created automatically on first upload; you can also create it yourself). Uploads go through the app (so the signed cookie identifies the owner) and are capped to keep the free 512 MB Render instance safe:

- 10 MB per file, 20 files and 50 MB total per visitor (all configurable)
- Download links expire after 5 minutes
- Text/code files (.txt .md .csv .json .py .js …) up to 200 KB can be attached to a chat message

Run `supabase_render_files.sql` once in Supabase to create the `render_files` table.

## Environment variables

Set these in the Render dashboard (**Service → Environment**). Secrets use `sync: false` in `render.yaml`, so Render asks for them but they are never stored in the repo. **Never commit API keys.**

### Required

| Variable | What it is |
| --- | --- |
| `APP_SECRET_KEY` | Long random string used to sign the anonymous user cookie. |
| `SUPABASE_URL` | Your Supabase project URL. |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service-role key (server only). Also used for file storage. |
| `ENVIRONMENT` | Set to `production` (already in `render.yaml`). Render also sets `RENDER=true`, which has the same effect. Needed so the user cookie is Secure. |
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

### Models, files and behavior (all optional)

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model ID. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model ID. |
| `OPENROUTER_MODEL` | `openrouter/free` | OpenRouter model ID. |
| `XKIRO_MODEL` | first free model | Default xKiro chat model. Must be a free model. |
| `XKIRO_IMAGE_MODEL` | first free image model (`sensenova/sensenova-u1.5-lite`) | Default xKiro image model. Must be a free model. |
| `XKIRO_BASE_URL` | `https://api.xkiro.com/v1` | xKiro API base URL. |
| `CLOUDFLARE_IMAGE_MODEL` | `@cf/black-forest-labs/flux-2-klein-4b` | Cloudflare image model. |
| `IMAGE_DEFAULT_PROVIDER` | `cloudflare` | Image engine selected by default: `cloudflare` or `xkiro`. |
| `SUPABASE_FILES_BUCKET` | `render-files` | Private Supabase Storage bucket for uploads. |
| `FILE_MAX_SIZE` | `10485760` (10 MB) | Max size of one upload, in bytes. |
| `FILE_USER_MAX_FILES` | `20` | Max files per visitor (0 = unlimited). |
| `FILE_USER_MAX_TOTAL` | `52428800` (50 MB) | Max total storage per visitor, in bytes (0 = unlimited). |
| `AI_TIMEZONE` | `America/New_York` | Time zone given to the AI for the current date/time. |
| `OPENROUTER_SITE_URL` | empty | Sent to OpenRouter as the referer. |
| `OPENROUTER_APP_NAME` | `Render AI` | Sent to OpenRouter as the app title. |
| `SECURE_COOKIES` | `false` | Force Secure cookies (already implied on Render). |
| `PYTHON_VERSION` | `3.12.3` | Python version used by Render. |

`UPSTASH_BLOB_TOKEN` and `BLOB_MAX_FILE_SIZE` are no longer used (files moved to Supabase Storage); you can delete them from Render.

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

## xKiro

[xKiro](https://xkiro.com) is an OpenAI-compatible gateway. The app loads its public model list and keeps only models marked `free`; paid models are rejected by the server even if someone sends the ID by hand. If xKiro is rate-limited or out of free allowance, chat falls back to OpenRouter or Groq when those keys are set. xKiro images are asynchronous, so the Image Generator shows progress and keeps checking (up to 5 minutes).

## Supabase

Tables used: `render_users`, `render_usage`, `render_files`, `chat_users`, `chat_sessions`, `chat_messages`. If your `render_usage` table restricts which `provider` values are allowed, add `xkiro` and `xkiro_image`.

## Staying small (Render free = 512 MB RAM)

No front-end framework or chart library (donuts are inline SVG), a single static HTML file, short-lived caches for model lists, uploads capped at 10 MB and read once, generated images are never stored on the server, and chat memory lives in the browser.

## Run locally

```bash
pip install -r requirements.txt
export APP_SECRET_KEY=change-me
export SUPABASE_URL=...
export SUPABASE_SERVICE_ROLE_KEY=...
export XKIRO_API_KEY=...   # plus any other provider keys you want
uvicorn main:app --reload
```
