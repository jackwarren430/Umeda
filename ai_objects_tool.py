"""
ai_objects_tool.py

Generate a Pygame *objects module* from an image.
The module must define:
    def create_game(screen, background_surf, context): -> game_obj
and game_obj exposes: handle_event(event), update(dt), draw(screen)

It will be used by game_shell.py, which:
- blits the background for you (call draw for overlays/objects)
- provides audio helpers via context["audio"] (SAMPLE_RATE, init_audio, make_sound_from_wave)

Usage:
    from ai_objects_tool import generate_game_objects_module
    module_path = generate_game_objects_module("canvas.png", gameplay_hint="It's a piano...")

Requires:
    pip install --upgrade openai
Env:
    export OPENAI_API_KEY=sk-...
"""

from __future__ import annotations
import os, re, base64, mimetypes
from pathlib import Path
from datetime import datetime
from typing import Optional
from openai import OpenAI, BadRequestError


def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt if mt in ("image/png", "image/jpeg") else "image/png"

def _extract_any_text(resp) -> str:
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
    return text.strip() if "def create_game(" in text or "import pygame" in text else None


# === Prompt with contracts to avoid common Pygame errors ===

_AUDIO_CONTRACT = """
AUDIO CONTRACT (MUST FOLLOW)
- Use ONLY pygame and numpy. Do not load external audio files.
- Synthesis helpers come from context["audio"]:
    SAMPLE_RATE = context["audio"]["SAMPLE_RATE"]
    make_sound_from_wave = context["audio"]["make_sound_from_wave"]
- When you need a tone:
    t = np.linspace(0, dur, int(SAMPLE_RATE*dur), endpoint=False)
    wave = 0.6*np.sin(2*np.pi*freq*t)  # add short fade in/out to avoid clicks
    snd = make_sound_from_wave(wave, volume=0.8)
    snd.play()
"""

_DEFAULT_USER_INSTRUCTIONS = f"""
Generate ONLY a Python *module* that defines:

    def create_game(screen, background_surf, context) -> obj

The returned obj must implement:
    handle_event(event), update(dt), draw(screen)

Rules:
- Use ONLY pygame and numpy.
- Treat 'background_surf' as the already-blitted background; draw your overlays in draw(screen).
- Detect obvious interactive regions from the image (e.g., keys on a piano, labeled buttons).
- On mouse click, trigger appropriate actions (e.g., play tones for keys) using context['audio'] helpers.
- ESC or window close quits (handled by the shell); you only handle interactions.
- Put everything in ONE file; no extra text outside a single ```python``` block.

{_AUDIO_CONTRACT}

Your goal is to make the object fully interactable. For example, generate code to make a piano playable. 
If there is an "on" button on the piano, it should not play until the it is clicked. 
If there are unmarked buttons, sliders, etc., use your best judgement to give them some function as well.
For example, changing the tune to the piano sounds, the speed of a fan, etc. 
If the object should have obvious physical characteristics, such as a bouncy ball, give it physics, like gravity.
Another example would be a UFO, which should hover and move when dragged, and display some colored lights when clicked.

"""

def generate_game_objects_module(image_path: str,
                                 user_hint: str = "",
                                 out_dir: str | None = "games",
                                 base_name: str = "objects") -> str:
    """
    Ask GPT-5 to generate a game-objects module from an image.
    Returns a file path like ./games/objects_YYYYmmdd_HHMMSS.py
    """
    p = Path(image_path)
    if not p.is_file():
        raise FileNotFoundError(f"No file at: {image_path}")

    mime = _guess_mime(image_path)
    with open(image_path, "rb") as f:
        b64_input = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64_input}"

    prompt = _DEFAULT_USER_INSTRUCTIONS
    if user_hint:
        prompt += "\n\nSpecific description of the object:\n" + user_hint

    client = OpenAI()
    try:
        resp = client.responses.create(
            model="gpt-5",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
            tool_choice="none",
        )
    except BadRequestError as e:
        raise RuntimeError(f"OpenAI request failed: {getattr(e, 'message', str(e))}") from e

    text = _extract_any_text(resp)
    code = _extract_python_code(text)
    if not code:
        raise RuntimeError("LLM returned no python code block.\nFirst 400 chars:\n" + text[:400])

    base_dir = Path(__file__).resolve().parent
    out_dir_path = (Path(out_dir) if out_dir is not None else base_dir / "games")
    if not out_dir_path.is_absolute():
        # default relative to project root (same dir as this file)
        out_dir_path = base_dir / out_dir_path
    out_dir_path.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = out_dir_path / f"{base_name}_{stamp}.py"
    script_path.write_text(code + "\n", encoding="utf-8")
    return str(script_path)
