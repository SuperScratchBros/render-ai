"""Small runtime compatibility fixes for external provider APIs.

KiosAPI model discovery now lives in main.py, so this adapter only preserves the
NVIDIA image compatibility handler for deployments using this entrypoint.
"""
import time

import httpx
from fastapi import HTTPException

import main


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
            return {"url": result, "model": model}

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
