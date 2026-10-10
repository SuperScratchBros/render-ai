""""Provider compatibility shims for the Render entrypoint.

KiosAPI is an OpenAI-compatible gateway. Its live model catalog is loaded server-side
so the UI can offer every text/chat model exposed by the configured KiosAPI key.
"""
import base64
import os
import time

import httpx
from fastapi import HTTPException

import main


KIOSAPI_DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
_KIOSAPI_CACHE = {"at": 0.0, "models": []}
_KIOSAPI_CACHE_TTL = 300


def kiosapi_settings():
    key = main.clean_key(os.getenv("KIOSAPI_API_KEY", ""))
    model = (os.getenv("KIOSAPI_MODEL", KIOSAPI_DEFAULT_MODEL).strip() or KIOSAPI_DEFAULT_MODEL)
    base_url = (os.getenv("KIOSAPI_BASE_URL", "https://api.kiosapi.id/v1").strip() or "https://api.kiosapi.id/v1").rstrip("/")
    return key, model, base_url


def install_kiosapi():
    """Register KiosAPI as a server-side OpenAI-compatible chat provider."""
    key, model, base_url = kiosapi_settings()
    main.MODELS["kiosapi"] = {
        "label": "KiosAPI",
        "model": model,
        "url": f"{base_url}/chat/completions",
        "key": "KIOSAPI_API_KEY",
    }
    main.PROVIDER_META["kiosapi"] = ("KiosAPI", "chat")
    main.DEFAULT_LIMITS["kiosapi"] = (200, 6000)
    main.USER_DAILY_LIMITS["kiosapi"] = 15
    main.USER_PER_MINUTE_LIMITS["kiosapi"] = 3
    main.MODELS["kiosapi"]["configured"] = bool(key)


def kiosapi_is_chat_model(item):
    """Exclude KiosAPI media/embedding/reranking endpoints from the chat picker."""
    model_id = item.get("id", "")
    if not isinstance(model_id, str) or not model_id.strip():
        return False
    model_id = model_id.strip().lower()

    # If the model metadata specifies a non-chat task, don't put it in chat.
    for field in ("task", "modality", "type", "capability"):
        value = item.get(field)
        if isinstance(value, str):
            task = value.lower().replace("-", "_").replace(" ", "_")
            if any(term in task for term in (
                "embedding", "rerank", "image_generation", "text_to_image",
                "video_generation", "text_to_video", "text_to_speech",
                "speech_to_text", "transcription", "music_generation",
                "ocr", "document_ai",
            )):
                return False
    endpoints = item.get("supported_endpoints")
    if isinstance(endpoints, list) and endpoints:
        normalized = [str(x).lower() for x in endpoints]
        if not any("chat" in x or "completion" in x for x in normalized):
            return False

    non_chat_markers = (
        "embedding", "rerank", "ranker", "-ocr", "/ocr", "document-ai",
        "native/", "gpt-image", "/imagen-", "imagen-3", "nano-banana",
        "imagine-image", "wan-2.7-image", "image-01", "glm-image",
        "/veo-", "video", "seedance", "lyria", "music-", "/tts",
        "speech-", "-tts", "-stt", "-asr", "transcribe", "whisper",
    )
    return not any(marker in model_id for marker in non_chat_markers)


def kiosapi_catalog(force=False):
    """Read KiosAPI's authenticated /models list; cache it for five minutes."""
    key, default_model, base_url = kiosapi_settings()
    if not key:
        return []
    now = time.time()
    if not force and _KIOSAPI_CACHE["models"] and now - _KIOSAPI_CACHE["at"] < _KIOSAPI_CACHE_TTL:
        return _KIOSAPI_CACHE["models"]

    try:
        response = httpx.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            timeout=15,
        )
        if response.status_code == 200:
            payload = response.json()
            raw_models = payload.get("data", []) if isinstance(payload, dict) else []
            models = []
            seen = set()
            for item in raw_models:
                if not isinstance(item, dict) or not kiosapi_is_chat_model(item):
                    continue
                model_id = item["id"].strip()
                if model_id in seen:
                    continue
                seen.add(model_id)
                models.append({
                    "id": model_id,
                    "label": model_id,
                    "provider": "KiosAPI",
                    "configured": True,
                    "access_tier": item.get("access_tier") or item.get("tier"),
                })
            if models:
                models.sort(key=lambda item: item["id"].casefold())
                _KIOSAPI_CACHE.update(at=now, models=models)
                return models
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        pass

    # Prefer a stale catalog if available. If the API's catalog is temporarily unavailable,
    # retain the documented default model as a small safety net rather than hiding KiosAPI.
    if _KIOSAPI_CACHE["models"]:
        return _KIOSAPI_CACHE["models"]
    return [{
        "id": default_model,
        "label": default_model,
        "provider": "KiosAPI",
        "configured": True,
    }]


_original_resolve_model = main.resolve_model
def resolve_model_with_kiosapi(provider, requested=None):
    if provider != "kiosapi":
        return _original_resolve_model(provider, requested)
    key, default_model, _ = kiosapi_settings()
    if not key:
        raise HTTPException(503, "KiosAPI is not configured. Add KIOSAPI_API_KEY in Render.")
    requested = (requested or "").strip() or default_model
    allowed = {item["id"] for item in kiosapi_catalog()}
    if requested not in allowed:
        raise HTTPException(400, "Choose a chat model from the current KiosAPI model list.")
    return requested


_original_resolve_selected_chat_model = main.resolve_selected_chat_model
def resolve_selected_chat_model_with_kiosapi(selection):
    if isinstance(selection, str) and selection.startswith("kiosapi:"):
        requested = selection.split(":", 1)[1].strip()
        if not requested:
            raise HTTPException(400, "Invalid KiosAPI model selection.")
        if not kiosapi_settings()[0]:
            raise HTTPException(503, "KiosAPI is not configured. Add KIOSAPI_API_KEY in Render.")
        return "kiosapi", requested
    return _original_resolve_selected_chat_model(selection)


def config_with_kiosapi():
    data = _original_config()
    key, default_model, _ = kiosapi_settings()
    # Remove the single generic KiosAPI row and replace it with model-specific entries.
    data["models"] = [item for item in data.get("models", []) if item.get("id") != "kiosapi"]
    catalog = kiosapi_catalog() if key else []
    if catalog:
        data["models"].extend(catalog_entry for catalog_entry in catalog)
    else:
        data["models"].append({
            "id": "kiosapi:" + default_model,
            "label": default_model,
            "model": default_model,
            "provider": "KiosAPI",
            "configured": bool(key),
        })
    data.setdefault("features", {})["kiosapi"] = bool(key)
    return data


install_kiosapi()
main.resolve_model = resolve_model_with_kiosapi
main.resolve_selected_chat_model = resolve_selected_chat_model_with_kiosapi
_original_config = main.config

# FastAPI stores the endpoint callable on the route's dependant at registration time, so
# replace both references for /api/config to make the live catalog visible in the browser.
for route in main.app.routes:
    if getattr(route, "path", None) == "/api/config" and hasattr(route, "dependant"):
        route.endpoint = config_with_kiosapi
        route.dependant.call = config_with_kiosapi
        break


def generate_nvidia_image(prompt, user_id):
    """Generate FLUX.2 Klein 4B through NVIDIA's OpenAI-compatible image API."""
    key = main.clean_key(main.os.getenv("NVIDIA_API_KEY", ""))
    if not key:
        raise HTTPException(503, "NVIDIA image generation is not configured. Add NVIDIA_API_KEY in Render.")
    if len(prompt) > 4000:
        raise HTTPException(400, "Image prompts can be at most 4000 characters.")

    model = main.NVIDIA_IMAGE_MODEL
    url = f"{main.NVIDIA_BASE_URL}/images/generations"
    payload = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "response_format": "b64_json",
    }

    last_detail = "NVIDIA image generation failed."
    for attempt in range(3):
        try:
            response = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
        except httpx.HTTPError as exc:
            if attempt < 2:
                time.sleep(attempt + 1)
                continue
            raise HTTPException(502, "Could not reach NVIDIA image generation.") from exc

        if response.status_code in {200, 201}:
            try:
                data = response.json()
            except ValueError as exc:
                raise HTTPException(502, "NVIDIA returned invalid image data.") from exc

            items = data.get("data") if isinstance(data, dict) else None
            item = items[0] if isinstance(items, list) and items else {}
            image_b64 = item.get("b64_json") if isinstance(item, dict) else None
            image_url = item.get("url") if isinstance(item, dict) else None

            if isinstance(image_b64, str) and image_b64:
                if image_b64.startswith("data:image/") and ";base64," in image_b64:
                    result = image_b64
                else:
                    result = f"data:image/png;base64,{image_b64}"
            elif isinstance(image_url, str) and image_url:
                result = image_url
            else:
                raise HTTPException(502, "NVIDIA returned no generated image data.")

            main.record_usage(user_id, "nvidia_image", "image", model)
            return result

        try:
            body = response.json()
            last_detail = str(body.get("detail") or body.get("message") or body.get("error") or response.text)[:500]
        except (ValueError, AttributeError):
            last_detail = (response.text or "NVIDIA image generation failed.")[:500]

        if response.status_code in {408, 425, 429, 500, 502, 503, 504} and attempt < 2:
            time.sleep(attempt + 1)
            continue
        if response.status_code in {401, 403}:
            raise HTTPException(502, "NVIDIA rejected the image API key. Check NVIDIA_API_KEY in Render.")
        raise HTTPException(502, f"NVIDIA image generation failed (HTTP {response.status_code}): {last_detail}")

    raise HTTPException(502, last_detail)


# /api/image resolves generate_nvidia_image from main.py's module globals at request time.
main.generate_nvidia_image = generate_nvidia_image
app = main.app
