"""
ai_router.py — Router Step 1 (Contract chooser)

What it does
------------
Given:
  • an image path (PNG/JPG)
  • a short natural-language description from the user
  • a catalog of contract options (id + title + when)

It asks GPT-5 to choose the best contract and returns STRICT JSON:
  {
    "contract_id": "<one of the provided ids>",
    "confidence": 0.0-1.0,
    "reason": "short rationale",
    "assumptions": ["optional", "brief", "assumptions"]
  }

Usage
-----
Python:
    from ai_router import select_contract, DEFAULT_CONTRACTS
    choice = select_contract("drawing.png", "This is a piano keyboard.")
    print(choice)

CLI:
    python ai_router.py ./drawing.png --desc "This is a piano keyboard." --out choice.json

Requirements
-----------
    pip install --upgrade openai
    export OPENAI_API_KEY=sk-...

Notes
-----
- Returns a Python dict; raises RuntimeError on invalid/ambiguous responses.
- You can pass a custom list of contracts (list[dict]) or use DEFAULT_CONTRACTS below.
"""

from __future__ import annotations
import os, json, base64, mimetypes, argparse, sys, re
from typing import List, Dict, Any, Optional
from openai import OpenAI, BadRequestError

# ---- Default contract catalog (short & unambiguous) ----
DEFAULT_CONTRACTS: List[Dict[str, str]] = [
    {"id": "functional_object",     "title": "Functional Object",     "when": "Instrument or control panel; discrete actions; no physics"},
    {"id": "physical_object",       "title": "Physical Object",       "when": "Single movable thing; gravity/bounce/drag"},
    {"id": "character_controller",  "title": "Character Controller",  "when": "Controllable player sprite; keyboard input"},
    {"id": "vehicle_arcade",        "title": "Vehicle (Arcade)",      "when": "Car/boat/plane top-down steering"},
    {"id": "pattern_sequencer",     "title": "Pattern/Sequencer",     "when": "Grid pads/steps; timed clock/tempo"},
    {"id": "puzzle_grid",           "title": "Puzzle/Grid",           "when": "Board tiles with rules; win/lose"},
    {"id": "path_follower",         "title": "Path/Follower",         "when": "Road/track/arrow path; sprite follows"},
    {"id": "particle_emitter",      "title": "Particle/Emitter",      "when": "Sparks/stars/clouds; clicks spawn bursts"},
    {"id": "growth_animation",      "title": "Growth/Stateful Anim.", "when": "Plant/progress/timeline stages"},
    {"id": "shooter_projectile",    "title": "Shooter/Projectile",    "when": "Aim and fire projectiles at targets"},
]

# ---- Prompt pieces ----
CHOOSER_SYSTEM = (
    "You are a strict router. You ONLY return compact JSON that matches the provided schema. "
    "Choose the single best contract from the catalog based on the image and description."
)

CHOOSER_USER_TEMPLATE = """Choose exactly ONE contract.

Inputs:
- Description: {user_desc}
- Contract catalog (array of objects: id, title, when):
{catalog_json}

Output JSON schema (strict):
{{
  "contract_id": "<one of the catalog ids>",
  "confidence": <number between 0 and 1>,
  "reason": "1-2 short sentences why this contract fits",
  "assumptions": ["optional short assumptions (if any)"]
}}

Return ONLY valid JSON. No code, no markdown, no explanations.
"""

# ---- Utilities ----
def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt if mt in ("image/png", "image/jpeg") else "image/png"

def _to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{_guess_mime(path)};base64,{b64}"

def _extract_text_from_response(resp) -> str:
    """
    Robustly pull plain text out of a Responses API object.
    Prefer resp.output_text; otherwise flatten message content.
    """
    # Try the helper if present
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt

    # Fallback: walk output items
    chunks = []
    for item in getattr(resp, "output", []) or []:
        t = getattr(item, "type", None)
        if t in ("message", "output_text"):
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    ctext = getattr(c, "text", None)
                    if isinstance(ctext, str):
                        chunks.append(ctext)
    return "\n".join(chunks).strip()

def _coerce_json(s: str) -> Dict[str, Any]:
    """
    Try to parse strict JSON. If the model wraps it with extra text,
    extract the first {...} block heuristically.
    """
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        # Heuristic: extract the first top-level {} block
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            raise RuntimeError("Router returned no JSON.")
        return json.loads(m.group(0))

def _validate_choice(obj: Dict[str, Any], catalog_ids: List[str]) -> Dict[str, Any]:
    req = ("contract_id", "confidence")
    for k in req:
        if k not in obj:
            raise RuntimeError(f"Router JSON missing required key: {k}")

    if obj["contract_id"] not in catalog_ids:
        raise RuntimeError(f"contract_id '{obj['contract_id']}' not in catalog ids {catalog_ids}")

    conf = obj["confidence"]
    if not (isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0):
        raise RuntimeError("confidence must be a number between 0 and 1")

    # Optional fields normalization
    obj.setdefault("reason", "")
    obj.setdefault("assumptions", [])
    if not isinstance(obj["assumptions"], list):
        obj["assumptions"] = [str(obj["assumptions"])]

    # Make types tidy
    obj["confidence"] = float(conf)
    obj["reason"] = str(obj["reason"])
    obj["assumptions"] = [str(x) for x in obj["assumptions"]]
    return obj

# ---- Public API ----
def select_contract(
    image_path: str,
    user_description: str,
    contracts: Optional[List[Dict[str, str]]] = None,
    model: str = "gpt-5",
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Run Router Step 1. Returns a dict:
      {"contract_id": "...", "confidence": 0.87, "reason": "...", "assumptions": [...]}
    Raises RuntimeError on failure.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"No file at: {image_path}")
    catalog = contracts or DEFAULT_CONTRACTS
    catalog_ids = [c["id"] for c in catalog]

    data_url = _to_data_url(image_path)
    user_msg = CHOOSER_USER_TEMPLATE.format(
        user_desc=user_description or "(none provided)",
        catalog_json=json.dumps(catalog, ensure_ascii=False, indent=2),
    )

    client = OpenAI()
    try:
        resp = client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": CHOOSER_SYSTEM},
                    {"type": "input_text", "text": user_msg},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
            # No tools; we want pure JSON text
            tool_choice="none",
        )
    except BadRequestError as e:
        raise RuntimeError(f"OpenAI request failed: {getattr(e, 'message', str(e))}") from e

    text = _extract_text_from_response(resp)
    obj = _coerce_json(text)
    obj = _validate_choice(obj, catalog_ids)
    return obj

# ---- CLI ----
def _main():
    ap = argparse.ArgumentParser(description="Router Step 1: choose a gameplay contract from an image.")
    ap.add_argument("image", help="Path to PNG/JPG drawing")
    ap.add_argument("--desc", default="", help="Short description (e.g., 'This is a piano keyboard')")
    ap.add_argument("--catalog", default="", help="Path to a custom catalog JSON (list of {id,title,when})")
    ap.add_argument("--out", default="", help="Write the JSON decision to a file")
    ap.add_argument("--model", default="gpt-5", help="Model name")
    ap.add_argument("--temp", type=float, default=0.1, help="Sampling temperature (default 0.1)")
    args = ap.parse_args()

    contracts = None
    if args.catalog:
        with open(args.catalog, "r", encoding="utf-8") as f:
            contracts = json.load(f)

    try:
        choice = select_contract(args.image, args.desc, contracts=contracts, model=args.model, temperature=args.temp)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)

    text = json.dumps(choice, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Wrote decision to {args.out}")
    else:
        print(text)

if __name__ == "__main__":
    _main()
