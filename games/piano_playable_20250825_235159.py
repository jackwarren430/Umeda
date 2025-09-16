import sys
import math
import pygame
import numpy as np

# -------------- Audio synthesis --------------

SAMPLE_RATE = 44100

def init_audio():
    try:
        pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=512)
    except Exception:
        pass
    pygame.mixer.init()

_tone_cache = {}

def note_freq_from_c4(semitones_from_c4):
    return 261.6255653005986 * (2 ** (semitones_from_c4 / 12.0))

NOTE_TO_ST = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4,
    'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

def get_tone(freq, duration=0.5, volume=0.6):
    key = (round(freq, 3), round(duration, 3), round(volume, 3))
    if key in _tone_cache:
        return _tone_cache[key]
    n_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    # Simple sine with quick fade-in/out to avoid clicks
    wave = np.sin(2 * np.pi * freq * t)
    attack = int(0.01 * n_samples) or 1
    release = int(0.05 * n_samples) or 1
    env = np.ones_like(wave)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    wave = wave * env * volume
    audio = (wave * 32767).astype(np.int16)
    snd = pygame.sndarray.make_sound(audio)
    _tone_cache[key] = snd
    return snd

# -------------- Image analysis (simple heuristics) --------------

def surface_to_gray_and_mask(surface, black_thresh=80):
    arr = pygame.surfarray.array3d(surface)  # shape (w, h, 3)
    arr = arr.astype(np.float32)
    # Convert to (h, w) grayscale
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).T
    mask_black = gray < black_thresh
    return gray, mask_black

def max_run_length_1d(bool_row):
    max_run = run = 0
    for v in bool_row:
        if v:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return max_run

def cluster_indices(indices, tol=3):
    if not indices:
        return []
    indices = sorted(indices)
    clusters = []
    cur = [indices[0]]
    for v in indices[1:]:
        if v - cur[-1] <= tol:
            cur.append(v)
        else:
            clusters.append(cur)
            cur = [v]
    clusters.append(cur)
    # Use the average index of each cluster
    return [int(round(sum(c) / len(c))) for c in clusters]

def find_keyboard_rect(mask_black, w, h):
    # 1) Find strong horizontal lines by long black runs per row
    row_scores = [max_run_length_1d(mask_black[y, :]) for y in range(h)]
    candidates = [y for y, s in enumerate(row_scores) if s > 0.5 * w]
    candidates = [y for y in candidates if int(h * 0.15) < y < int(h * 0.9)]
    line_rows = cluster_indices(candidates, tol=2)

    best = None
    # consider all pairs and choose a plausible key-band
    for i in range(len(line_rows)):
        for j in range(i + 1, len(line_rows)):
            y1, y2 = line_rows[i], line_rows[j]
            if y2 - y1 < int(h * 0.1) or y2 - y1 > int(h * 0.5):
                continue
            band = mask_black[y1:y2, :]
            density = band.mean() if band.size else 0
            score = density * (y2 - y1)
            if best is None or score > best[0]:
                best = (score, y1, y2)

    if best is None:
        # Fallback: lower-middle band guess
        y2 = int(h * 0.72)
        y1 = int(h * 0.52)
    else:
        _, y1, y2 = best

    kb_h = y2 - y1
    # 2) Find left/right borders using vertical runs inside this band; avoid outer 10% of image width
    x_start = max(0, int(w * 0.08))
    x_end = min(w, int(w * 0.92))
    col_scores = []
    for x in range(x_start, x_end):
        col = mask_black[y1:y2, x]
        col_scores.append(max_run_length_1d(col))

    left = None
    right = None
    threshold = int(kb_h * 0.8)
    for idx, s in enumerate(col_scores):
        if s >= threshold:
            left = x_start + idx
            break
    for idx in range(len(col_scores) - 1, -1, -1):
        if col_scores[idx] >= threshold:
            right = x_start + idx
            break

    if left is None or right is None or right - left < int(w * 0.2):
        # Fallback to central area
        left = int(w * 0.18)
        right = int(w * 0.82)

    return pygame.Rect(left, y1, max(1, right - left), max(1, y2 - y1))

def detect_black_key_rects(mask_black, kb_rect):
    # Detect filled dark vertical bars in the upper part of the keyboard area
    y1 = kb_rect.top
    y2 = kb_rect.bottom
    x1 = kb_rect.left
    x2 = kb_rect.right
    kb_h = kb_rect.height
    band_top = y1 + int(0.06 * kb_h)
    band_bottom = y1 + int(0.70 * kb_h)
    band_h = max(1, band_bottom - band_top)

    cols_ratio = []
    for x in range(x1, x2):
        col = mask_black[band_top:band_bottom, x]
        cols_ratio.append(col.mean())

    # Threshold by mean + deviation heuristic
    arr = np.array(cols_ratio, dtype=np.float32)
    thr = max(0.35, float(arr.mean() + 0.5 * arr.std()))
    binary = arr > thr

    # Find contiguous segments as potential black keys
    rects = []
    run_start = None
    for i, v in enumerate(binary):
        if v and run_start is None:
            run_start = i
        elif not v and run_start is not None:
            run_end = i
            width = run_end - run_start
            if width >= int(kb_rect.width * 0.025):
                bx1 = x1 + run_start
                bw = width
                rects.append(pygame.Rect(bx1, y1, bw, band_h))
            run_start = None
    if run_start is not None:
        run_end = len(binary)
        width = run_end - run_start
        if width >= int(kb_rect.width * 0.025):
            bx1 = x1 + run_start
            bw = width
            rects.append(pygame.Rect(bx1, y1, bw, band_h))

    # Merge very close rects (thin gaps)
    merged = []
    for r in rects:
        if not merged:
            merged.append(r)
        else:
            last = merged[-1]
            if r.left - last.right <= int(kb_rect.width * 0.01):  # tiny gap -> merge
                new = last.union(r)
                merged[-1] = pygame.Rect(new.left, y1, new.width, band_h)
            else:
                merged.append(r)

    # Filter to at most 5 with heuristic: keep the 5 widest; else fallback later
    merged.sort(key=lambda R: R.width, reverse=True)
    merged = merged[:5]
    merged.sort(key=lambda R: R.centerx)

    return merged

# -------------- Piano model --------------

class PianoKey:
    def __init__(self, rect, name, color):
        self.rect = rect
        self.name = name   # e.g., 'C', 'C#'
        self.color = color # 'white' or 'black'
        self.freq = note_freq_from_c4(NOTE_TO_ST[name])
        self.hover = False
        self.active = False

    def play(self):
        snd = get_tone(self.freq, duration=0.45, volume=0.65 if self.color == 'white' else 0.55)
        snd.play()

def build_piano(kb_rect, black_rects_detected):
    keys_white = []
    keys_black = []

    # White keys: 7 equal divisions across the keyboard band
    white_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    sw = kb_rect.width / 7.0
    for i, nm in enumerate(white_names):
        r = pygame.Rect(int(kb_rect.left + i * sw), kb_rect.top, int(sw + 0.999), kb_rect.height)
        keys_white.append(PianoKey(r, nm, 'white'))

    # Black keys:
    black_names_order = ['C#', 'D#', 'F#', 'G#', 'A#']
    blacks = []

    if len(black_rects_detected) == 5:
        # Use detected positions
        black_rects_detected.sort(key=lambda R: R.centerx)
        for r, nm in zip(black_rects_detected, black_names_order):
            # Slightly narrow and center them visually
            shrink = int(r.width * 0.15)
            r = pygame.Rect(r.left + shrink, r.top, max(2, r.width - 2 * shrink), int(kb_rect.height * 0.7))
            blacks.append((r, nm))
    else:
        # Fallback: synthesize black keys as centered between adjacent white keys (except E-F gap)
        idx_pairs = [(0,1), (1,2), (3,4), (4,5), (5,6)]
        for (i_left, i_right), nm in zip(idx_pairs, black_names_order):
            wl = keys_white[i_left].rect
            wr = keys_white[i_right].rect
            cx = (wl.right + wr.left) // 2
            bw = int(sw * 0.6)
            bx = int(cx - bw // 2)
            by = kb_rect.top
            bh = int(kb_rect.height * 0.65)
            blacks.append((pygame.Rect(bx, by, bw, bh), nm))

    for r, nm in blacks:
        keys_black.append(PianoKey(r, nm, 'black'))

    return keys_white, keys_black

# -------------- Rendering helpers --------------

def draw_hover_overlay(surface, rect, color, alpha):
    overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    overlay.fill((*color, alpha))
    surface.blit(overlay, rect.topleft)

def draw_outline(surface, rect, color, width=2, alpha=180):
    overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(overlay, (*color, alpha), pygame.Rect(0, 0, rect.width, rect.height), width)
    surface.blit(overlay, rect.topleft)

# -------------- Main loop --------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py path/to/piano_image.png")
        return

    pygame.init()
    init_audio()

    img_path = sys.argv[1]
    try:
        bg = pygame.image.load(img_path)
    except Exception as e:
        print("Failed to load image:", e)
        return

    w, h = bg.get_width(), bg.get_height()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Interactive Piano (click keys, ESC to quit, S to screenshot)")

    # Analyze image to locate keyboard area and keys
    _, mask_black = surface_to_gray_and_mask(bg)
    kb_rect = find_keyboard_rect(mask_black, w, h)
    black_detected = detect_black_key_rects(mask_black, kb_rect)
    white_keys, black_keys = build_piano(kb_rect, black_detected)

    clock = pygame.time.Clock()
    running = True

    def key_at_pos(pos):
        # Prioritize black keys (they sit above whites visually)
        for k in black_keys:
            if k.rect.collidepoint(pos):
                return k
        for k in white_keys:
            if k.rect.collidepoint(pos):
                return k
        return None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    pygame.image.save(screen, "screenshot.png")
                    print("Saved screenshot.png")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                k = key_at_pos(event.pos)
                if k:
                    print(f"Played {k.name} ({int(round(k.freq))} Hz)")
                    k.active = True
                    k.play()
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                for k in white_keys + black_keys:
                    k.active = False

        # Hover states
        mx, my = pygame.mouse.get_pos()
        for k in white_keys + black_keys:
            k.hover = k.rect.collidepoint((mx, my))

        # Draw
        screen.blit(bg, (0, 0))

        # Subtle indication of interactive keyboard area
        draw_outline(screen, kb_rect.inflate(6, 6), (50, 150, 255), width=3, alpha=100)

        # Draw overlays for hover/active
        # Whites first
        for k in white_keys:
            if k.hover or k.active:
                draw_hover_overlay(screen, k.rect, (255, 230, 80), 60 if k.hover and not k.active else 110)
        # Blacks on top
        for k in black_keys:
            if k.hover or k.active:
                draw_hover_overlay(screen, k.rect, (120, 200, 255), 80 if k.hover and not k.active else 140)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
