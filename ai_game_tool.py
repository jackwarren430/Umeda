"""
ai_game_tool.py

Generate a complete Pygame game script from a canvas image using GPT-5.
- Sends your image (as a data URL) + instructions via the Responses API.
- Extracts a single ```python ...``` code block and writes it to a temp file.
- Returns the path to that temp *.py script.

Usage:
    from ai_game_tool import generate_pygame_from_image
    script_path = generate_pygame_from_image("canvas.png")

Requirements:
    pip install --upgrade openai
Env:
    export OPENAI_API_KEY=sk-...
"""

from __future__ import annotations
import os, re, base64, mimetypes, tempfile
from typing import Optional
from openai import OpenAI, BadRequestError

from pathlib import Path
from datetime import datetime


def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt if mt in ("image/png", "image/jpeg") else "image/png"


def _extract_any_text(resp) -> str:
    # Robustly flatten text from Responses API
    texts = []
    for item in getattr(resp, "output", []) or []:
        t = getattr(item, "type", None)
        if t in ("message", "output_text"):
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    tx = getattr(c, "text", None)
                    if isinstance(tx, str):
                        texts.append(tx)
    if not texts:
        tx = getattr(resp, "output_text", None)
        if isinstance(tx, str):
            texts.append(tx)
    return "\n".join(texts)


_CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_CODE_BLOCK_ANY = re.compile(r"```(.*?)```", re.DOTALL)


def _extract_python_code(text: str) -> Optional[str]:
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _CODE_BLOCK_ANY.search(text)
    if m:
        return m.group(1).strip()
    # fallback: if it looks like python, just return the text
    return text.strip() if "import pygame" in text or "def main(" in text else None


_DEFAULT_SYSTEM = (
    "You write clean, runnable Python games using Pygame. "
    "You respond with ONE python code block only, nothing else."
)


_DEFAULT_USER_INSTRUCTIONS = """
Generate a COMPLETE, runnable Pygame script that turns the provided image
(sys.argv[1]) into an interactive game. Use ONLY pygame and numpy. Follow these rules:

GENERAL
- Load the background from sys.argv[1]; size the window to that image.
- Detect obvious interactive regions (e.g., for a piano: white/black keys). Mouse click triggers the action.
- When the user clicks/presses on an interactive region:
    * If it's a piano keyboard: play the corresponding musical tone (use pygame.sndarray + numpy
      to synthesize a sine wave; no external audio files).
    * If it's not a piano: make the region do something sensible (toggle, print, play a simple tone).
- Show a subtle hover highlight on interactive elements.
- Controls: mouse click to interact; ESC or window close to quit.
- No network calls. No extra file I/O except optional screenshots if you include a key.
- ESC or window close quits. Put everything in one file with `if __name__ == "__main__": main()`.
- Return ONLY one fenced Python code block (no extra text).

You must implement create_game, as seen below:

def create_game(screen, background_surf, context) -> obj
# obj has: handle_event(event), update(dt), draw(screen)
# Use context['audio']['make_sound_from_wave'] for tones if needed.



INTERACTION EXAMPLE (PIANO)
- If the image looks like a keyboard, segment white keys into 7 equal regions and black keys between them.
- On click: synthesize tones with `numpy` + the helper above. No external audio files.
- Show hover highlight over interactive regions.

DELIVERABLE
- One Python code block only. No explanations. Must run with: `python your_script.py path/to/image.png`
""".strip()


def generate_pygame_from_image(image_path: str,
                               gameplay_hint: str = "",
                               out_dir: str | None = None,
                               base_name: str = "gpt_playable") -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"No file at: {image_path}")
    """ 
    Create a Pygame script via GPT-5 that makes the image playable.
    :param image_path: path to PNG/JPG to send as context.
    :param gameplay_hint: optional extra instruction (e.g., "It's a piano—map white keys to C-D-E-F-G-A-B").
    :return: path to a temporary .py script file.
    """
    mime = _guess_mime(image_path)
    with open(image_path, "rb") as f:
        b64_input = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64_input}"

    user_prompt = _DEFAULT_USER_INSTRUCTIONS
    if gameplay_hint:
        user_prompt += "\n\nAdditional hint from user:\n" + gameplay_hint

    client = OpenAI()

    try:
        resp = client.responses.create(
            model="gpt-5",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
            # No tools: we want pure code as text
            tool_choice="none",
        )
    except BadRequestError as e:
        raise RuntimeError(f"OpenAI request failed: {getattr(e, 'message', str(e))}") from e

    text = _extract_any_text(resp)
    code = _extract_python_code(text)
    if not code:
        # dump a small excerpt for debugging
        raise RuntimeError("LLM returned no python code block.\nFirst 400 chars:\n" + text[:400])

    base_dir = Path(__file__).resolve().parent
    out_dir_path = Path(out_dir) if out_dir else (base_dir / "games")
    out_dir_path.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = out_dir_path / f"{base_name}_{stamp}.py"

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)
        f.write("\n")

    return str(script_path)

        