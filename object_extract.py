"""
object_extract.py
Extract the main foreground object from a drawing and save it as a cropped RGBA PNG
with transparency. Returns the sprite path and its original location on the canvas.

Usage (Python):
    from object_extract import extract_main_object_to_png
    sprite_path, meta = extract_main_object_to_png("canvas.png")  # meta: {"x","y","w","h"}

CLI:
    python object_extract.py canvas.png --out games/sprites/sprite.png --debug debug.png

Notes:
- Works well for line-drawn shapes on a light/white background (your PyQt canvas).
- Picks the *largest external contour*, preferring contours not glued to the image border.
- If nothing is found, raises RuntimeError (your caller already treats extraction as optional).
- Deps: pip install opencv-python numpy
"""

from __future__ import annotations
import os
from pathlib import Path
import tempfile
from typing import Tuple, Dict, Optional

import cv2
import numpy as np


def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Auto-select Canny thresholds based on image median."""
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


def _touches_border(rect: Tuple[int, int, int, int], w: int, h: int, pad: int = 2) -> bool:
    x, y, rw, rh = rect
    return x <= pad or y <= pad or (x + rw) >= (w - pad) or (y + rh) >= (h - pad)


def _best_external_contour(contours, w: int, h: int):
    """
    Score contours by area, penalize those touching borders (often entire frame/box).
    Return the best contour; None if empty list.
    """
    best = None
    best_score = -1.0
    for c in contours:
        area = float(cv2.contourArea(c))
        if area <= 0:
            continue
        x, y, rw, rh = cv2.boundingRect(c)
        score = area
        if _touches_border((x, y, rw, rh), w, h):
            score *= 0.6  # penalize border-touching shapes
        if score > best_score:
            best_score = score
            best = c
    return best


def extract_main_object_to_png(
    bg_path: str,
    out_path: Optional[str] = None,
    min_area_ratio: float = 0.0025,  # ignore tiny specks < 0.25% of image
    debug_out: Optional[str] = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Detect the main object and save as a cropped RGBA sprite with transparency.

    :param bg_path: Input PNG/JPG path (your canvas snapshot).
    :param out_path: Optional output sprite PNG path. Defaults to "<bg_stem>_sprite.png".
    :param min_area_ratio: Minimum contour area relative to image area to accept.
    :param debug_out: Optional path to write a debug visualization (PNG).
    :return: (sprite_path, {"x": int, "y": int, "w": int, "h": int})
    """
    if not os.path.isfile(bg_path):
        raise FileNotFoundError(bg_path)

    print(f"[object-extract] extracting main object from {bg_path}")

    bgr = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {bg_path}")
    H, W = bgr.shape[:2]
    print(f"[object-extract] canvas size: {W}x{H}")

    # 1) Grayscale + mild blur to stabilize edges
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2) Edge detection + dilation to thicken line art
    edges = _auto_canny(gray, sigma=0.33)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # 3) Find external contours
    cnts_info = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]

    if not contours:
        raise RuntimeError("No contours found")

    best = _best_external_contour(contours, W, H)
    if best is None:
        raise RuntimeError("No suitable contour found")

    x, y, rw, rh = cv2.boundingRect(best)
    area = rw * rh
    print(f"[object-extract] best contour rect={(x, y, rw, rh)} area={area}")
    if area < min_area_ratio * (W * H):
        raise RuntimeError("Detected object is too small; likely noise")

    # 4) Build filled mask for the best contour; close small holes
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(mask, [best], -1, 255, thickness=cv2.FILLED)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # 5) Compose RGBA and crop
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask  # alpha channel

    crop = rgba[y:y + rh, x:x + rw].copy()
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        raise RuntimeError("Empty crop after contour extraction")

    # 6) Write sprite
    if out_path is None:
        p = Path(bg_path)
        out_path = str(p.with_name(p.stem + "_sprite.png"))
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(out_path, crop)
    if not ok:
        raise RuntimeError(f"Failed to write sprite PNG: {out_path}")

    # Optional: write debug overlay
    if debug_out:
        dbg = bgr.copy()
        cv2.rectangle(dbg, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
        cv2.drawContours(dbg, [best], -1, (255, 0, 0), 2)
        alpha_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        alpha_color = (0.3 * alpha_color + 0.7 * dbg).astype(np.uint8)
        cv2.imwrite(debug_out, alpha_color)

    meta = {"x": int(x), "y": int(y), "w": int(rw), "h": int(rh)}
    print(f"[object-extract] sprite saved to {out_path}")
    print(f"[object-extract] meta={meta}")
    return out_path, meta


def extract_component_from_bbox(
    bg_path: str,
    bbox: Dict[str, int],
    out_path: str,
    min_area_ratio: float = 0.0025,
) -> Tuple[str, Dict[str, int]]:
    """
    Crop a rectangular region and attempt to extract the foreground object within it.

    Falls back to a plain rectangular crop if contour extraction fails.
    Returns the sprite path and absolute coordinates (x, y, w, h) relative to the
    original background image.
    """
    if not os.path.isfile(bg_path):
        raise FileNotFoundError(bg_path)
    if not bbox:
        raise ValueError("bbox is required")

    x = int(bbox.get("x", 0))
    y = int(bbox.get("y", 0))
    w = int(bbox.get("w", 0))
    h = int(bbox.get("h", 0))
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid bbox dimensions: {bbox}")

    print(f"[component-extract] bbox={bbox} bg={bg_path}")

    bgr = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {bg_path}")

    H, W = bgr.shape[:2]
    x0 = max(0, min(W - 1, x))
    y0 = max(0, min(H - 1, y))
    x1 = max(0, min(W, x0 + w))
    y1 = max(0, min(H, y0 + h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"BBox outside image bounds: {bbox}")

    crop = bgr[y0:y1, x0:x1]
    fd, temp_path = tempfile.mkstemp(prefix="component_", suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(temp_path, crop)
        candidate_path = out_path
        out_dir = Path(candidate_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            sprite_path, meta = extract_main_object_to_png(
                temp_path,
                out_path=candidate_path,
                min_area_ratio=min_area_ratio,
            )
            meta = {
                "x": x0 + int(meta["x"]),
                "y": y0 + int(meta["y"]),
                "w": int(meta["w"]),
                "h": int(meta["h"]),
            }
            print(f"[component-extract] contour success path={sprite_path} meta={meta}")
            return sprite_path, meta
        except RuntimeError:
            print("[component-extract] contour failed, using raw crop")
            rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = 255
            ok = cv2.imwrite(candidate_path, rgba)
            if not ok:
                raise RuntimeError(f"Failed to write component PNG: {candidate_path}")
            meta = {
                "x": x0,
                "y": y0,
                "w": x1 - x0,
                "h": y1 - y0,
            }
            print(f"[component-extract] fallback path={candidate_path} meta={meta}")
            return candidate_path, meta
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ---------- CLI for quick testing ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Extract main object to a cropped RGBA sprite PNG.")
    ap.add_argument("image", help="Input PNG/JPG path")
    ap.add_argument("--out", default="", help="Output sprite path (PNG). Default: <stem>_sprite.png")
    ap.add_argument("--debug", default="", help="Optional debug visualization path (PNG)")
    ap.add_argument("--min-area", type=float, default=0.0025, help="Min area ratio to accept (default 0.0025)")
    args = ap.parse_args()

    sprite, info = extract_main_object_to_png(
        args.image,
        out_path=(args.out or None),
        min_area_ratio=args.min_area,
        debug_out=(args.debug or None),
    )
    print("Sprite:", sprite)
    print("Meta:", info)
