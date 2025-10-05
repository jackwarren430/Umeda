"""
ai_fulfill.py — Router Step 2 (Contract fulfillment)

Given:
  • chosen contract_id from Router Step 1
  • image path and optional user hint
  • optional sprite metadata from your extractor

Builds a strict contract prompt (incl. Display/Audio contracts) and asks GPT-5
to generate a SMALL Python module that integrates with game_shell.py.

Deliverable module REQUIREMENT (enforced in prompt):
    def create_game(screen, background_surf, context) -> obj
Where obj implements:
    handle_event(event)    # per pygame.event.get()
    update(dt: float)      # dt in seconds
    draw(screen)           # background is already blitted by the shell

Usage:
    from ai_fulfill import fulfill_contract
    module_path = fulfill_contract(
        image_path="canvas.png",
        contract_id="physical_object",
        user_hint="This is a bouncy ball",
        sprite_meta={"path":"games/sprites/ball.png","x":120,"y":260,"w":64,"h":64}
    )

Requirements:
    pip install --upgrade openai
    export OPENAI_API_KEY=sk-...
"""

from __future__ import annotations
import os, re, json, base64, mimetypes
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from openai import OpenAI, BadRequestError


# ---------- Shared environment contracts injected into every prompt ----------

DISPLAY_CONTRACT = """
DISPLAY CONTRACT (MUST FOLLOW)
- The boiler has already created the display and blits the background each frame.
- Do NOT call pygame.display.set_mode() or Surface.convert()/convert_alpha() on the background.
- You will receive:
    screen: pygame.Surface for drawing
    background_surf: pygame.Surface (already blitted by the shell before your draw())
- Your draw(screen) should render overlays/objects on top of the already-blitted background.
"""

AUDIO_CONTRACT = """
AUDIO CONTRACT (MUST FOLLOW)
- Allowed libraries: ONLY pygame and numpy.
- Use audio helpers from context["audio"]:
    SAMPLE_RATE = context["audio"]["SAMPLE_RATE"]
    make_sound_from_wave = context["audio"]["make_sound_from_wave"]
- To play a tone:
    import numpy as np
    t = np.linspace(0, dur, int(SAMPLE_RATE*dur), endpoint=False)
    wave = 0.6*np.sin(2*np.pi*freq*t) * np.linspace(1, 0.9, t.size)  # tiny fade
    snd = make_sound_from_wave(wave, volume=0.8)
    snd.play()
"""

API_INTERFACE = """
MODULE INTERFACE (MUST IMPLEMENT)
- Provide EXACTLY ONE function:
    def create_game(screen, background_surf, context):
        '''
        Returns an object with methods:
          handle_event(event)    # called for each event
          update(dt)             # dt: float seconds
          draw(screen)           # draw your overlays/sprites; background is already drawn
        '''
- No main loop, no file/network I/O, no blocking calls, no external dependencies beyond pygame/numpy.
- The module must fit in ONE file and be deterministic and readable.
- Return ONLY one fenced python code block. No prose.
"""

# ---------- Contract catalog (add more as you grow) ----------

CONTRACT_SPECS: Dict[str, Dict[str, Any]] = {

    # Instruments, panels, UI surfaces (e.g., keyboard, drum pads, sliders)
    "functional_object": {
        "title": "Functional Object",
        "summary": "Discrete interactive regions; no physics. Good for instruments/UIs (piano keys, pads, sliders).",
        "requirements": """
BEHAVIOR
- Detect clear interactive regions from the image (e.g., white/black piano keys or buttons).
- Hover highlight + click activation.
- For a keyboard: map white keys to C D E F G A B (one octave), black keys to sharps (#).
- Use context['audio'] helpers to synthesize tones; keep it lightweight.
- Optional: small HUD text for help ('Click keys • ESC to quit').

INPUTS & CONTEXT
- If context['sprite'] exists, treat it as a decorative overlay unless useful for hit tests.
- Use screen.get_size() as the world bounds; do not change display.

CONTROLS
- Mouse hover/click for interaction.
- 'R' to reset any internal state.

PERFORMANCE
- Keep per-frame logic O(nRegions) and optimized enough for 60 FPS.
""",
    },

    # A single sprite with simple physics (gravity, bounce, drag)
    "physical_object": {
        "title": "Physical Object",
        "summary": "One main object with basic physics (velocity, gravity, bounce, drag).",
        "requirements": """
BEHAVIOR
- Use context['sprite']['surface'] as the visible object and context['sprite']['rect'] for its position.
- Simulate vx, vy with gravity toggle, restitution (bounce), and air drag.
- Collide with window edges; bounce or clamp sensibly.
- Mouse drag: click+drag moves the sprite; on release, impart a small velocity from drag delta.

INPUTS & CONTEXT
- REQUIRED: context['sprite'] exists with keys {surface, rect, meta}.
- Do not modify pixel data; move the rect and draw the surface.
- Keep the object's initial position for 'R' reset.

CONTROLS
- Arrow keys nudge the sprite.
- 'G' toggles gravity (on/off); 'R' resets; 'P' pauses physics.

PERFORMANCE
- Keep update O(1); use dt integration (semi-implicit Euler is fine).
""",
    },

    # Player-controlled object (WASD/arrow movement), no gravity by default
    "character_controller": {
        "title": "Character Controller",
        "summary": "Controllable player sprite; acceleration/deceleration; screen bounds.",
        "requirements": """
BEHAVIOR
- Must use context['sprite']
- Keyboard input (WASD/Arrows) controls acceleration; cap max speed.
- Optional: dash (Shift) with cooldown.

CONTROLS
- WASD/Arrows to move; Shift to dash; 'R' reset.

COLLISIONS
- Clamp to window bounds; no world physics required.
""",
    },

    # Clocked grid (pads/steps), simple sequencer
    "pattern_sequencer": {
        "title": "Pattern/Sequencer",
        "summary": "Grid of steps/pads with a clock (BPM); click to toggle cells; play tones as the playhead advances.",
        "requirements": """
BEHAVIOR
- Derive a simple MxN grid overlay aligned to a detected panel/area.
- Maintain a playhead that advances with BPM (default 120).
- Activated cells play tones using context['audio'].

CONTROLS
- Click to toggle cells; Space to start/stop; Up/Down to adjust BPM; 'C' clear; 'R' reset.

PERFORMANCE
- Keep per-tick work O(activeCells).
""",
    },

    "particle_emitter": {
        "title": "Particle Emitter",
        "summary": "Sprite-based particle system with continuous emission and click bursts; supports gravity, drag, pooling.",
        "requirements": """
    BEHAVIOR
    - If context['sprite'] exists:
      • Set the emitter origin to sprite['rect'].center at start.
      • Derive a small particle image from sprite['surface'] (keep alpha; scale so the max dimension ~24 px).
    - If context['sprite'] is missing, synthesize a round 16–24 px RGBA particle (soft edge).
    - Implement continuous emission when 'emitting' is ON, plus click-burst at the mouse position.
    - Each particle tracks: position (float x,y), velocity (vx,vy), remaining_life (seconds), and an alpha/scale factor.
    - Update per dt:
      • Apply gravity (toggleable) and linear air drag.
      • Fade alpha to 0 by end of life; kill when life <= 0.
    - Draw particles on top of the already-blitted background. Do not modify the background surface.

    INPUTS & CONTEXT
    - Respect the Display/Audio contracts (no display init, no file/network I/O).
    - If context['audio'] is present, you MAY play a very short click on bursts using make_sound_from_wave (optional).

    CONTROLS
    - Mouse left click: set emitter origin to mouse position and spawn a burst.
    - Space: toggle continuous emission on/off.
    - G: toggle gravity.
    - R: reset (clear all particles; move origin back to initial sprite center if available).
    - '[' / ']': decrease/increase emission rate.
    - Up/Down: decrease/increase base particle speed.
    - Left/Right: decrease/increase spread angle (cone width).

    PERFORMANCE
    - Use a fixed-size pool or free-list (e.g., MAX_PARTICLES ≈ 1000). Reuse entries; avoid per-frame allocations.
    - Precompute the particle Surface once (from the sprite or synthetic). Do NOT create/destroy Surfaces during update/draw.
    - Keep update O(N_live). Target 60 FPS.

    DELIVERABLE
    - Provide EXACTLY:
        def create_game(screen, background_surf, context) -> obj
      where obj implements:
        handle_event(event)    # route mouse/keyboard controls
        update(dt: float)      # integrate physics and emit new particles
        draw(screen)           # blit particles over the background
    - Use only pygame (and numpy only if you synthesize audio).
    - Return ONE fenced python code block. No extra prose.
    """,
    },

}

# ---------- Prompt builder ----------

SYSTEM_MSG = (
    "You write small, robust Pygame modules that plug into a provided shell. "
    "You must obey the interface and environment constraints exactly."
)

FULFILL_USER_TEMPLATE = """Implement the "{title}" contract ({contract_id}).

Description from user: {user_desc}

Environment (provided by shell):
- pygame is already initialized; screen and background_surf are valid.
- The shell draws the background before calling your draw(screen).
- context contains:
    audio helpers (SAMPLE_RATE, make_sound_from_wave)
    image_path (string)
    sprite (optional): {{
        'surface': pygame.Surface with per-pixel alpha,
        'rect': pygame.Rect initial placement,
        'meta': {{'x': int, 'y': int, 'w': int, 'h': int}}
    }}
    components (optional list): [{{
        'id': str,
        'label': str,
        'role': str,
        'description': str,
        'surface': pygame.Surface,
        'rect': pygame.Rect,
        'meta': {{'x': int, 'y': int, 'w': int, 'h': int}}
    }}]
    graph (optional dict): {{'nodes': [...], 'edges': [...]}}

Follow ALL constraints below.

{DISPLAY_CONTRACT}

{AUDIO_CONTRACT}

{API_INTERFACE}

CONTRACT REQUIREMENTS
{contract_requirements}

DELIVERABLE
- Return ONLY one fenced python code block with the complete module. No extra text.
"""

# ---------- Helpers ----------

def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt if mt in ("image/png", "image/jpeg") else "image/png"

def _to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{_guess_mime(path)};base64,{b64}"

_CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_CODE_BLOCK_ANY = re.compile(r"```(.*?)```", re.DOTALL)

def _extract_python_code(text: str) -> Optional[str]:
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _CODE_BLOCK_ANY.search(text)
    if m:
        return m.group(1).strip()
    # fallback if model forgot fences
    return text.strip() if "def create_game(" in text else None

def _extract_any_text(resp) -> str:
    # Prefer unified output_text if available
    ot = getattr(resp, "output_text", None)
    if isinstance(ot, str) and ot.strip():
        return ot
    # Flatten messages
    chunks = []
    for item in getattr(resp, "output", []) or []:
        t = getattr(item, "type", None)
        if t in ("message", "output_text"):
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    tx = getattr(c, "text", None)
                    if isinstance(tx, str):
                        chunks.append(tx)
    return "\n".join(chunks).strip()

# ---------- Public API ----------

def build_fulfillment_prompt(contract_id: str, user_desc: str) -> str:
    spec = CONTRACT_SPECS.get(contract_id)
    if not spec:
        raise ValueError(f"Unknown contract_id: {contract_id}")
    return FULFILL_USER_TEMPLATE.format(
        title=spec["title"],
        contract_id=contract_id,
        user_desc=user_desc or "(none)",
        DISPLAY_CONTRACT=DISPLAY_CONTRACT.strip(),
        AUDIO_CONTRACT=AUDIO_CONTRACT.strip(),
        API_INTERFACE=API_INTERFACE.strip(),
        contract_requirements=spec["requirements"].strip(),
    )

def fulfill_contract(
    image_path: str,
    contract_id: str,
    user_hint: str = "",
    sprite_meta: Optional[Dict[str, Any]] = None,
    component_graph_yaml: Optional[str] = None,
    components: Optional[List[Dict[str, Any]]] = None,
    out_dir: str = "games",
    base_name: str = "objects",
    model: str = "gpt-5",
) -> str:
    """
    Generate the contract-specific module.
    Returns the path to ./games/<base_name>_YYYYmmdd_HHMMSS.py

    component_graph_yaml: raw YAML text describing the DAG (optional)
    components: list of component dicts with sprite metadata (optional)
    """
    p = Path(image_path)
    if not p.is_file():
        raise FileNotFoundError(f"No file at: {image_path}")

    prompt = build_fulfillment_prompt(contract_id, user_hint)
    data_url = _to_data_url(image_path)
    print(f"[fulfill] contract_id={contract_id} image={image_path}")
    print(f"[fulfill] user_hint='{user_hint}'")

    # Build input messages with image + optional sprite meta (as JSON text)
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": data_url},
    ]
    if sprite_meta:
        content.append({"type": "input_text", "text": "Sprite metadata (JSON): " + json.dumps(sprite_meta)})
        print(f"[fulfill] sprite_meta={sprite_meta}")
    if component_graph_yaml:
        content.append({"type": "input_text", "text": "Component graph (YAML):\n" + component_graph_yaml})
        preview = component_graph_yaml.splitlines()[:10]
        print("[fulfill] graph preview:")
        for line in preview:
            print("   ", line)
    if components:
        content.append({"type": "input_text", "text": "Component sprites (JSON): " + json.dumps(components)})
        print(f"[fulfill] components count={len(components)}")

    client = OpenAI()
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            tool_choice="none",  # text-only; we want a code block
        )
    except BadRequestError as e:
        raise RuntimeError(f"OpenAI request failed: {getattr(e, 'message', str(e))}") from e

    text = _extract_any_text(resp)
    code = _extract_python_code(text)
    if not code:
        raise RuntimeError("LLM returned no python code block.\nFirst 400 chars:\n" + text[:400])

    base_dir = Path(__file__).resolve().parent
    out_dir_path = (Path(out_dir) if Path(out_dir).is_absolute() else base_dir / out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = out_dir_path / f"{base_name}_{contract_id}_{stamp}.py"
    script_path.write_text(code + "\n", encoding="utf-8")
    print(f"[fulfill] wrote module to {script_path}")
    return str(script_path)
