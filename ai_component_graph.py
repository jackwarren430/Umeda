"""
ai_component_graph.py — Generate a DAG of sub-components from an image + contract context.

Returns YAML (as text) describing nodes (with bounding boxes) and directed edges.
"""

from __future__ import annotations
import base64
import mimetypes
import os

from openai import BadRequestError, OpenAI


SYSTEM_PROMPT = (
    "You analyze drawings and plan interactive components. "
    "Respond ONLY with YAML describing a directed acyclic graph (DAG) of components, "
    "including bounding boxes within the reference image."
)

USER_TEMPLATE = """Image-driven planning task.

Inputs:
- Contract ID: {contract_id}
- Contract summary: {contract_summary}
- Contract requirements:
{contract_requirements}
- User hint: {user_hint}

Produce graph.yaml (YAML only, no prose) with the following schema:
	nodes:
	  - id: short slug (lowercase, hyphen separated)
	    label: short human name
	    role: high-level role (e.g., control, indicator, connector)
	    description: 1-2 sentences describing the sub-component
	    bbox:
	      x: integer (top-left pixel)
	      y: integer
	      w: integer (width)
	      h: integer (height)
	edges:
	  - from: source node id
	    to: target node id
	    relation: short verb phrase (e.g., activates, powers)

Constraints:
- The graph MUST be a DAG. No cycles.
- Include every sub-component needed to fulfill the contract.
- Bounding boxes must align with the actual regions in the provided image (use the image reference).
- Use at least one edge if multiple nodes exist.
- Omit empty arrays entirely if there are no edges.

Output the YAML directly with no surrounding commentary.
"""


def _to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    mime, _ = mimetypes.guess_type(path)
    if mime not in ("image/png", "image/jpeg"):
        mime = "image/png"
    return f"data:{mime};base64,{b64}"


def generate_component_graph(
    image_path: str,
    contract_id: str,
    contract_summary: str,
    contract_requirements: str,
    user_hint: str = "",
    model: str = "gpt-5",
) -> str:
    """Return YAML text describing the component DAG."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    client = OpenAI()
    user_msg = USER_TEMPLATE.format(
        contract_id=contract_id,
        contract_summary=contract_summary.strip() if contract_summary else "(none)",
        contract_requirements=contract_requirements.strip(),
        user_hint=user_hint or "(no hint)",
    )

    print("[component-graph] generating graph")
    print(f"[component-graph] image={image_path}")
    print(f"[component-graph] contract_id={contract_id}")
    print(f"[component-graph] model={model}")

    content = [
        {"type": "input_text", "text": SYSTEM_PROMPT},
        {"type": "input_text", "text": user_msg},
        {"type": "input_image", "image_url": _to_data_url(image_path)},
    ]

    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            tool_choice="none",
        )
    except BadRequestError as e:
        raise RuntimeError(f"OpenAI request failed: {getattr(e, 'message', str(e))}") from e

    text = getattr(resp, "output_text", None)
    print("[component-graph] response received")
    if isinstance(text, str) and text.strip():
        preview = text.strip().splitlines()
        print("[component-graph] yaml preview:")
        for line in preview[:10]:
            print("   ", line)
        return text.strip()

    collected = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) in ("message", "output_text"):
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    txt = getattr(c, "text", None)
                    if isinstance(txt, str):
                        collected.append(txt)
    if not collected:
        raise RuntimeError("Graph generation returned no YAML text")
    text = "\n".join(collected).strip()
    preview = text.splitlines()
    print("[component-graph] yaml preview (merged):")
    for line in preview[:10]:
        print("   ", line)
    return text


__all__ = ["generate_component_graph"]
