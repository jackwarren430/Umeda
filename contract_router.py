"""
contract_router.py — Router Step 1 (contract selection)

Usage (CLI):
    export OPENAI_API_KEY=sk-...
    python contract_router.py ./drawing.png "simple description of the drawing"

Programmatic:
    from contract_router import route_contract, default_catalog
    result = route_contract("drawing.png", "a simple piano keyboard")
    print(result)  # {'contract_id': 'functional_object', 'confidence': 0.92, ...}

This module sends ONLY:
  - the image (as a data URL),
  - the user's short description,
  - the contract catalog (ids + short 'when' descriptions)

and asks GPT-5 to respond with a STRICT JSON object:
  {
    "contract_id": "<one of the ids in catalog>",
    "confidence": 0.0..1.0,
    "reason": "short reason",
    "assumptions": ["optional", "list"]
  }

No code generation happens here. Step 2 will use the chosen contract.
"""

from __future__ import annotations
import os, json, base64, mimetypes, sys
from typing import List, Dict, Any, Optional
from openai import OpenAI, BadRequestError

# ---------- Catalog ----------

def default_catalog() -> List[Dict[str, str]]:
    """
    Minimal, high-signal catalog. You can extend/replace at call time.
    Each item must have: {'id': str, 'ti
