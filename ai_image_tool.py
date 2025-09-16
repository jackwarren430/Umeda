"""
Generate an image with GPT-5 (Responses API + image_generation tool)
from a prompt + a local reference image (PNG/JPG).

Usage:
    from ai_image_tool import generate_image_with_gpt5
    out_png = generate_image_with_gpt5("drawing.png",
                                       "Clean this into crisp, even line art. Keep the same layout.")

Reqs:  pip install --upgrade openai
Env:   export OPENAI_API_KEY=sk-...
"""

from __future__ import annotations
import os, base64, mimetypes, tempfile
from typing import Optional
from openai import OpenAI, BadRequestError




def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt if mt in ("image/png", "image/jpeg") else "image/png"


def _extract_image_b64(resp) -> Optional[str]:
    """
    Robustly pull a base64 image out of a Responses API result.
    Handles:
      - output items with type='image' (image.base64 / image.b64_json)
      - ImageGenerationCall items with .result (base64 PNG)
      - fallback resp.data[0].b64_json
    """
    # Preferred: iterate the new .output list
    for item in getattr(resp, "output", []) or []:
        # Case 1: old/alt shape: explicit image item
        t = getattr(item, "type", None)
        if t == "image":
            img = getattr(item, "image", None)
            if img:
                b64 = getattr(img, "base64", None) or getattr(img, "b64_json", None)
                if b64:
                    return b64

        # Case 2: image tool call object (e.g., ImageGenerationCall) with .result
        if hasattr(item, "result") and isinstance(getattr(item, "result"), str):
            return item.result  # already base64 (often starts with 'iVBORw0...')

        # Case 3: sometimes image nested in content list
        content = getattr(item, "content", None)
        if isinstance(content, list):
            for c in content:
                if getattr(c, "type", None) == "image":
                    img = getattr(c, "image", None)
                    if img:
                        b64 = getattr(img, "base64", None) or getattr(img, "b64_json", None)
                        if b64:
                            return b64

    # Fallback: legacy Images-like payloads
    data = getattr(resp, "data", None)
    if data and len(data):
        b64 = getattr(data[0], "b64_json", None)
        if b64:
            return b64

    return None



def generate_image_with_gpt5(image_path: str, prompt: str) -> str:
    """
    Call GPT-5 with the image_generation tool using your local image + prompt.
    Returns a path to a temporary PNG.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"No file at: {image_path}")

    mime = _guess_mime(image_path)
    with open(image_path, "rb") as f:
        b64_input = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64_input}"

    client = OpenAI()

    try:
        resp = client.responses.create(
            model="gpt-5",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    # IMPORTANT: use image_url (data URL), not "image"
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
            tools=[{"type": "image_generation"}],
            #tool_choice={"type": "auto"},
        )
    except BadRequestError as e:
        # Surface the server’s message (super helpful when shapes drift)
        raise RuntimeError(f"OpenAI request failed: {getattr(e, 'message', str(e))}") from e

    # After getting `resp`
    for i, item in enumerate(getattr(resp, "output", []) or []):
        print(i, item.__class__.__name__, getattr(item, "type", None), hasattr(item, "result"))


    b64 = _extract_image_b64(resp)
    
    if not b64:
        # Optional: uncomment to debug the raw response shape
        #print("RAW RESPONSE:", resp)
        raise RuntimeError("No image found in model response.")


    img_bytes = base64.b64decode(b64)
    fd, out_path = tempfile.mkstemp(prefix="gpt5_image_", suffix=".png")

    with os.fdopen(fd, "wb") as f:
        f.write(img_bytes)

    return out_path


if __name__ == "__main__":
    # quick manual test
    path = "./drawing.png"
    prompt = "Generate a cleaned up version of this image. Keep everything the same, just make the lines straight (or curvy if needed) and the text look nice"
    out_png = generate_image_with_gpt5(path, prompt)
    print("Wrote:", out_png)
