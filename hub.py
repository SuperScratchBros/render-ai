"""NLGEP — one service for the home page, NLGEP AI and NLGEP Chat.

Start with:  uvicorn hub:app --host 0.0.0.0 --port $PORT

  /          home page (pick AI or Chat)
  /ai        NLGEP AI    (also /ai/, /ai/anything)
  /chat      NLGEP Chat  (also /chat/, /chat/anything)

Every page URL is served directly. Nothing here ever redirects to the home page, so a browser
such as NLGEP can load /ai or /chat on its own (with or without a trailing slash).
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.routing import Mount

import chat_backend
import main as ai_main  # the existing AI app, unchanged: we reuse its API routes

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

# redirect_slashes=False: "/chat/" must be served, never bounced to somewhere else.
app = FastAPI(title="NLGEP", redirect_slashes=False, docs_url=None, redoc_url=None, openapi_url=None)


# ---------------------------------------------------------------- pages

FX_HEAD = '<link rel="stylesheet" href="/static/nlgep-fx.css">'
FX_BODY = '<script src="/static/nlgep-fx.js" defer></script>'
_ai_cache = {"mtime": None, "html": ""}


def ai_html() -> str:
    """The AI page (index.html, untouched on disk) with the sound/hover files added when served."""
    f = BASE_DIR / "index.html"
    if not f.is_file():
        raise HTTPException(404, "The AI page file is missing.")
    mtime = f.stat().st_mtime
    if _ai_cache["mtime"] != mtime:
        html = f.read_text(encoding="utf-8")
        i = html.find("</head>")
        if i != -1:
            html = html[:i] + FX_HEAD + html[i:]
        j = html.rfind("</body>")
        html = html[:j] + FX_BODY + html[j:] if j != -1 else html + FX_BODY
        _ai_cache.update(mtime=mtime, html=html)
    return _ai_cache["html"]


def _head_ok() -> Response:
    return Response(status_code=200, headers={"Content-Type": "text/html; charset=utf-8"})


def home_page(request: Request, rest: str = ""):
    if request.method == "HEAD":
        return _head_ok()
    return FileResponse(STATIC_DIR / "home.html")


def chat_page(request: Request, rest: str = ""):
    if request.method == "HEAD":
        return _head_ok()
    return FileResponse(STATIC_DIR / "chat.html")


def ai_page(request: Request, rest: str = ""):
    if request.method == "HEAD":
        return _head_ok()
    return HTMLResponse(ai_html())


def add_pages(paths, handler):
    for p in paths:
        app.add_api_route(p, handler, methods=["GET", "HEAD"], include_in_schema=False)


add_pages(["/", "/home", "/home/"], home_page)
add_pages(["/chat", "/chat/", "/chat/{rest:path}"], chat_page)
add_pages(["/ai", "/ai/", "/ai/{rest:path}"], ai_page)


@app.api_route("/health", methods=["GET", "HEAD"], include_in_schema=False)
def health():
    return {"ok": True, "supabase_configured": bool(ai_main.SUPABASE_URL and ai_main.SUPABASE_SERVICE_ROLE_KEY),
            "chat_configured": chat_backend.supabase_configured()}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------- APIs

app.include_router(chat_backend.router)  # /api/chat-*, /api/staff/*

# Reuse every route of the AI app (/api/ask, /api/config, /api/usage, ...) except the ones
# this file already provides (its own "/" page, /health, docs, and its static mount).
_SKIP_PATHS = {"/", "/health", "/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"}
for _route in ai_main.app.routes:
    if isinstance(_route, Mount) or getattr(_route, "path", None) in _SKIP_PATHS:
        continue
    app.router.routes.append(_route)


@app.middleware("http")
async def cache_headers(request: Request, call_next):
    """Pages and static files are re-checked on every load; API answers are never cached."""
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    elif "cache-control" not in response.headers:
        response.headers["Cache-Control"] = "no-cache"
    return response
