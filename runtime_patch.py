"""Small runtime compatibility fixes for external provider APIs.

This module is the Render entrypoint so provider-specific fixes can be isolated
from the main application while upstream APIs change.
"""
import base64
import os
import time

import httpx
from fastapi import HTTPException

import main


def install_kiosapi():
    """Add Kiosapi's OpenAI-compatible gateway without exposing its key to the browser."""
    key = main.clean_key(os.getenv("KIOSAPI_API_KEY", ""))
    model = (os.getenv("KIOSAPI_MODEL", "deepseek/deepseek-v4-flash").strip() or "deepseek/deepseek-v4-flash")
    base_url = (os.getenv("KIOSAPI_BASE_URL", "https://api.kiosapi.id/v1").strip() or "https://api.kiosapi.id/v1").rstrip("/")

    main.MODELS["kiosapi"] = {
        "label": "KiosAPI: " + model,
        "model": model,
        "url": f"{base_url}/chat/completions",
        "key": "KIOSAPI_API_KEY",
    }
    main.PROVIDER_META["kiosapi"] = ("KiosAPI", "chat")
    # Keep Kiosapi inside the same fair-use accounting system as the other chat gateways.
    main.DEFAULT_LIMITS["kiosapi"] = (200, 6000)
    main.USER_DAILY_LIMITS["kiosapi"] = 15
    main.USER_PER_MINUTE_LIMITS["kiosapi"] = 3

    # Make the live /api/config response reflect whether the secret is present.
    # MODELS is intentionally mutated server-side; the actual API key is never returned.
    main.MODELS["kiosapi"]["configured"] = bool(key)


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


install_kiosapi()

# /api/image resolves generate_nvidia_image from main.py's module globals at request time.
main.generate_nvidia_image = generate_nvidia_image
app = main.app
