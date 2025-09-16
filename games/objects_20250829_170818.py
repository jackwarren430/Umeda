import math
import pygame
import numpy as np

# Single-file pygame + numpy module

def create_game(screen, background_surf, context):
    class PianoGame:
        def __init__(self, screen, bg_surf, context):
            self.screen = screen
            self.bg_surf = bg_surf
            self.w, self.h = bg_surf.get_size()
            self.audio = context.get("audio", {})
            self.SAMPLE_RATE = int(self.audio.get("SAMPLE_RATE", 44100))
            self.make_sound_from_wave = self.audio.get("make_sound_from_wave")
            self.power_on = False
            self.transpose = 0  # semitone offset controlled by the knob
            self.knob_angle = 0.0
            self.dragging_knob = False

            # Visual state for pressed keys
            self.active_keys = {}  # key_id -> remaining_time

            # Build dark map from the background to detect regions
            self.gray = self._surface_to_gray(self.bg_surf)
            self.dark = self.gray < 70  # boolean map of dark pixels

            # Detect main UI regions from the image
            self.keyboard_rect = self._detect_keyboard_rect()
            self.white_keys, self.black_keys = self._detect_keys(self.keyboard_rect)
            self.all_keys = self._combine_keys_for_notes(self.white_keys, self.black_keys)

            # On/off circular button (search upper area for a ring)
            self.on_center, self.on_radius = self._detect_on_button()
            if self.on_center is None:
                # Fallback: put it roughly near the word "on" in the header area
                self.on_center = (int(self.w * 0.52), int(self.h * 0.23))
                self.on_radius = int(min(self.w, self.h) * 0.05)

            # Knob at the plus sign on the right: detect cross, else fallback
            self.knob_center, self.knob_radius = self._detect_knob()
            if self.knob_center is None:
                self.knob_center = (int(self.w * 0.82), int(self.h * 0.24))
            if self.knob_radius is None:
                self.knob_radius = int(min(self.w, self.h) * 0.055)

            # Set base MIDI so leftmost key is C4 by default
            self.base_midi = 60  # C4

            # Visual resources
            pygame.font.init()
            self.font = pygame.font.SysFont(None, max(16, int(self.h * 0.03)))

        # ---------------- Audio ----------------
        def _midi_to_freq(self, midi):
            return 440.0 * (2.0 ** ((midi - 69) / 12.0))

        def _play_tone(self, midi, dur=0.5):
            if not self.power_on:
                return
            if self.make_sound_from_wave is None:
                return  # audio helper missing
            freq = self._midi_to_freq(midi + self.transpose)
            sr = self.SAMPLE_RATE
            dur = max(0.08, float(dur))
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            # Sine with small fade in/out to avoid clicks
            wave = 0.6 * np.sin(2 * np.pi * freq * t)
            fade = min(0.02, dur * 0.2)
            n_f = max(1, int(sr * fade))
            env = np.ones_like(wave)
            if n_f > 1:
                ramp = np.linspace(0, 1, n_f)
                env[:n_f] *= ramp
                env[-n_f:] *= ramp[::-1]
            wave = wave * env
            snd = self.make_sound_from_wave(wave, volume=0.8)
            snd.play()

        # ---------------- Detection helpers ----------------
        def _surface_to_gray(self, surf):
            arr = pygame.surfarray.array3d(surf)  # (w, h, 3), but note array3d returns (w,h,3)
            # Transpose to (h, w, 3)
            arr = np.transpose(arr, (1, 0, 2)).astype(np.float32)
            # Luma approximation
            gray = (0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]).astype(np.uint8)
            return gray

        def _detect_keyboard_rect(self):
            h, w = self.gray.shape
            dark = self.dark
            y0 = int(0.35 * h)
            y1 = int(0.85 * h)
            row_counts = dark[y0:y1, :].sum(axis=1)
            # Rows with long dark horizontal lines (inner keyboard frame)
            thresh_row = int(0.20 * w)
            ys = np.where(row_counts > thresh_row)[0]
            if len(ys) >= 2:
                y_top = y0 + ys[0]
                y_bot = y0 + ys[-1]
            else:
                # Fallback
                y_top = int(0.55 * h)
                y_bot = min(h - 10, y_top + int(0.22 * h))

            # Columns with long dark vertical lines (left/right inner keyboard frame)
            col_counts = dark[y_top:y_bot, :].sum(axis=0)
            thresh_col = int(0.20 * (y_bot - y_top))
            xs = np.where(col_counts > thresh_col)[0]
            if len(xs) >= 2:
                x_left = xs[0]
                x_right = xs[-1]
            else:
                x_left = int(0.12 * w)
                x_right = int(0.88 * w)

            # Clamp and ensure minimal size
            x_left = max(0, min(x_left, w - 10))
            x_right = max(x_left + 40, min(x_right, w - 1))
            y_top = max(0, min(y_top, h - 10))
            y_bot = max(y_top + 30, min(y_bot, h - 1))

            # Slightly inset from the border lines
            rect = pygame.Rect(x_left + 2, y_top + 2, (x_right - x_left) - 4, (y_bot - y_top) - 4)
            return rect

        def _group_indices(self, idx, max_gap=2):
            groups = []
            if len(idx) == 0:
                return groups
            start = idx[0]
            prev = idx[0]
            for i in idx[1:]:
                if i - prev <= max_gap:
                    prev = i
                else:
                    groups.append((start, prev))
                    start = i
                    prev = i
            groups.append((start, prev))
            return groups

        def _detect_separators(self, kb_rect):
            # Detect vertical key separators (thin dark lines spanning full height of keyboard)
            x0, y0, w, h = kb_rect.x, kb_rect.y, kb_rect.w, kb_rect.h
            dark = self.dark
            # Bottom band emphasizes full-height vertical lines (black keys don't reach bottom)
            yb0 = y0 + int(0.75 * h)
            yb1 = y0 + h
            yb0 = min(max(y0, yb0), y0 + h - 1)

            # Sliding 3-pixel window for robustness
            col_counts = []
            for x in range(x0 + 1, x0 + w - 1):
                c = dark[y0:yb1, (x - 1):(x + 2)].sum()
                cbottom = dark[yb0:yb1, (x - 1):(x + 2)].sum()
                col_counts.append((c, cbottom))
            col_counts = np.array(col_counts)
            # Thresholds: enough dark pixels through full height AND specifically near bottom
            thr_full = int(0.30 * h * 3)
            thr_bottom = int(0.18 * (yb1 - yb0) * 3)
            candidates = np.where((col_counts[:, 0] > thr_full) & (col_counts[:, 1] > thr_bottom))[0]
            # Convert to absolute x positions
            xs = (x0 + 1) + candidates
            # Group nearby columns as the same separator line
            groups = self._group_indices(xs, max_gap=3)
            separators = [int((a + b) / 2) for (a, b) in groups]
            # Make sure left and right edges are included
            if len(separators) < 3:
                # Fallback: subdivide into 8 or 12 white keys depending on width
                n_keys = 12 if w > 0.45 * self.w else 8
                step = w / n_keys
                separators = [int(x0 + i * step) for i in range(n_keys + 1)]
            else:
                # Clip to keyboard rect
                separators = [max(x0, min(x, x0 + w)) for x in separators]
                separators = sorted(list(set([x0] + separators + [x0 + w])))

            # Enforce strictly increasing and unique
            s2 = [separators[0]]
            for s in separators[1:]:
                if s > s2[-1] + 1:
                    s2.append(s)
            return s2

        def _detect_black_keys(self, kb_rect):
            x0, y0, w, h = kb_rect.x, kb_rect.y, kb_rect.w, kb_rect.h
            dark = self.dark
            top_h = int(0.6 * h)
            # Sliding 3-pixel window vertical counts in top region
            counts = []
            for x in range(x0 + 1, x0 + w - 1):
                ctop = dark[y0:(y0 + top_h), (x - 1):(x + 2)].sum()
                counts.append(ctop)
            counts = np.array(counts)
            thr_black = int(0.22 * top_h * 3)  # black keys create tall dark columns in top region
            cand = np.where(counts > thr_black)[0]
            xs = (x0 + 1) + cand
            groups = self._group_indices(xs, max_gap=3)
            blacks = []
            avg_w_est = max(6, int(w / 14))
            for (a, b) in groups:
                width = b - a + 1
                if width > int(1.4 * avg_w_est):  # filter out too-wide blobs (likely frame)
                    continue
                bx0 = max(x0 + 1, a - 1)
                bx1 = min(x0 + w - 2, b + 1)
                rect = pygame.Rect(bx0, y0 + 1, bx1 - bx0 + 1, int(0.62 * h))
                blacks.append(rect)
            # Merge overlapping/very close black key boxes
            blacks_sorted = sorted(blacks, key=lambda r: r.centerx)
            merged = []
            for r in blacks_sorted:
                if not merged:
                    merged.append(r)
                else:
                    last = merged[-1]
                    if r.left <= last.right + 2:  # very close, merge
                        new = last.union(r)
                        new.height = min(last.height, r.height)
                        merged[-1] = new
                    else:
                        merged.append(r)
            return merged

        def _detect_keys(self, kb_rect):
            # White key rectangles from separators
            separators = self._detect_separators(kb_rect)
            white_keys = []
            for i in range(len(separators) - 1):
                x0 = separators[i] + 1
                x1 = separators[i + 1] - 1
                if x1 - x0 < 4:
                    continue
                rect = pygame.Rect(x0, kb_rect.y + 1, x1 - x0, kb_rect.h - 2)
                white_keys.append(rect)

            # Black keys from top dark clusters
            black_keys = self._detect_black_keys(kb_rect)

            # Filter black keys to lie above the gaps between adjacent whites (typical layout)
            filtered_blacks = []
            if white_keys:
                for br in black_keys:
                    cx = br.centerx
                    # Find adjacent white keys around cx
                    left_idx = None
                    right_idx = None
                    for i in range(len(white_keys) - 1):
                        if white_keys[i].right <= cx <= white_keys[i + 1].left:
                            left_idx = i
                            right_idx = i + 1
                            break
                    if left_idx is not None and right_idx is not None:
                        filtered_blacks.append(br)
            else:
                filtered_blacks = black_keys

            return white_keys, filtered_blacks

        def _combine_keys_for_notes(self, white_keys, black_keys):
            # Create unified key list for note ordering from left to right (chromatic)
            keys = []
            kid = 0
            for r in white_keys:
                keys.append({"id": kid, "rect": r, "kind": "white"})
                kid += 1
            for r in black_keys:
                keys.append({"id": kid, "rect": r, "kind": "black"})
                kid += 1
            # Sort by x center to define chromatic order
            keys.sort(key=lambda k: (k["rect"].centerx, 0 if k["kind"] == "white" else 1))
            # Assign midi numbers in chromatic order starting from base_midi
            for i, k in enumerate(keys):
                k["midi"] = self.base_midi + i
            # Also keep separate ordered lists for hit-testing (black above white)
            self.hit_black = sorted([k for k in keys if k["kind"] == "black"], key=lambda k: k["rect"].centerx)
            self.hit_white = sorted([k for k in keys if k["kind"] == "white"], key=lambda k: k["rect"].centerx)
            return keys

        def _detect_on_button(self):
            # Search for a dark ring in the upper area (around the "on" circle)
            h, w = self.gray.shape
            dark = self.dark
            x_min = int(0.38 * w)
            x_max = int(0.64 * w)
            y_min = int(0.12 * h)
            y_max = int(0.42 * h)
            if x_min >= x_max or y_min >= y_max:
                return None, None

            best_score = 0.0
            best_c = None
            best_r = None
            # Coarse sampling to keep it light
            xs = range(x_min, x_max, max(4, int((x_max - x_min) / 24)))
            ys = range(y_min, y_max, max(4, int((y_max - y_min) / 16)))
            r_min = int(0.03 * min(w, h))
            r_max = int(0.10 * min(w, h))
            r_step = max(2, int((r_max - r_min) / 8))
            for cy in ys:
                for cx in xs:
                    for r in range(r_min, r_max + 1, r_step):
                        # Sample along the ring
                        n_samp = 36
                        hits = 0
                        for k in range(n_samp):
                            ang = (2 * math.pi * k) / n_samp
                            for dr in (-1, 0, 1):  # ring thickness
                                px = int(cx + (r + dr) * math.cos(ang))
                                py = int(cy + (r + dr) * math.sin(ang))
                                if 0 <= px < w and 0 <= py < h and dark[py, px]:
                                    hits += 1
                        score = hits / (n_samp * 3)
                        if score > best_score:
                            best_score = score
                            best_c = (cx, cy)
                            best_r = r
            if best_c is None or best_score < 0.35:
                return None, None
            return best_c, best_r

        def _detect_knob(self):
            # Find the plus sign in the top-right area by cross-like density
            h, w = self.gray.shape
            dark = self.dark
            xr0 = int(0.68 * w)
            xr1 = int(0.95 * w)
            yr0 = int(0.12 * h)
            yr1 = int(0.42 * h)
            xr0 = min(max(0, xr0), w - 1)
            xr1 = min(max(0, xr1), w - 1)
            yr0 = min(max(0, yr0), h - 1)
            yr1 = min(max(0, yr1), h - 1)
            if xr0 >= xr1 or yr0 >= yr1:
                return None, None

            col_counts = dark[yr0:yr1, xr0:xr1].sum(axis=0)
            row_counts = dark[yr0:yr1, xr0:xr1].sum(axis=1)
            cx_rel = int(np.argmax(col_counts))
            cy_rel = int(np.argmax(row_counts))
            cx = xr0 + cx_rel
            cy = yr0 + cy_rel
            # Validate by checking local cross density
            vert = dark[yr0:yr1, max(0, cx - 1):min(w, cx + 2)].sum()
            hori = dark[max(0, cy - 1):min(h, cy + 2), xr0:xr1].sum()
            if vert + hori < 50:
                return None, None
            radius = int(min(w, h) * 0.055)
            return (cx, cy), radius

        # ---------------- Event handling ----------------
        def handle_event(self, event):
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # Toggle power if clicking on 'on' button circle
                if self._point_in_circle(mx, my, self.on_center, self.on_radius):
                    self.power_on = not self.power_on
                    return

                # Start dragging knob if inside knob circle
                if self._point_in_circle(mx, my, self.knob_center, self.knob_radius + 6):
                    self.dragging_knob = True
                    self._update_knob_from_mouse(mx, my)
                    return

                # Key press: prioritize black keys
                key = self._hit_test_keys((mx, my))
                if key is not None:
                    self._handle_key_press(key)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging_knob = False

            elif event.type == pygame.MOUSEMOTION and self.dragging_knob:
                mx, my = event.pos
                self._update_knob_from_mouse(mx, my)

        def _update_knob_from_mouse(self, mx, my):
            cx, cy = self.knob_center
            dx, dy = mx - cx, my - cy
            ang = math.atan2(dy, dx)  # [-pi, pi]
            self.knob_angle = ang
            # Map angle to semitone transpose in [-12, +12]
            # Use -150..+150 degrees active range for better control
            deg = math.degrees(ang)
            # Wrap to [-180, 180]
            if deg > 180:
                deg -= 360
            if deg < -180:
                deg += 360
            deg = max(-150.0, min(150.0, deg))
            t = (deg + 150.0) / 300.0  # 0..1
            self.transpose = int(round(-12 + t * 24))

        def _point_in_circle(self, x, y, center, radius):
            cx, cy = center
            return (x - cx) ** 2 + (y - cy) ** 2 <= (radius ** 2)

        def _hit_test_keys(self, pos):
            x, y = pos
            # Black keys first (they sit on top)
            for k in self.hit_black:
                if k["rect"].collidepoint(x, y):
                    return k
            for k in self.hit_white:
                if k["rect"].collidepoint(x, y):
                    return k
            return None

        def _handle_key_press(self, key):
            # Visual flash
            self.active_keys[key["id"]] = 0.18  # highlight for ~0.18s
            # Play tone if powered on
            self._play_tone(key["midi"], dur=0.45)

        # ---------------- Update & Draw ----------------
        def update(self, dt):
            # Fade active key highlights
            to_del = []
            for k, t in self.active_keys.items():
                t -= dt
                if t <= 0:
                    to_del.append(k)
                else:
                    self.active_keys[k] = t
            for k in to_del:
                self.active_keys.pop(k, None)

        def draw(self, screen):
            # Draw overlays: key highlights, knob indicator, power state
            overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)

            # Key highlights
            for k in self.all_keys:
                kid = k["id"]
                if kid in self.active_keys:
                    t = self.active_keys[kid]
                    alpha = int(160 * max(0.0, min(1.0, t / 0.18)))
                    if k["kind"] == "white":
                        color = (50, 150, 255, alpha)
                    else:
                        color = (255, 230, 50, alpha)
                    pygame.draw.rect(overlay, color, k["rect"])
            # Draw keyboard hitboxes subtle outline (helps feedback)
            kb_color = (0, 0, 0, 60)
            for k in self.hit_white:
                pygame.draw.rect(overlay, kb_color, k["rect"], 1)
            for k in self.hit_black:
                pygame.draw.rect(overlay, kb_color, k["rect"], 1)

            # Power indicator fill inside the on-button circle
            if self.on_center and self.on_radius:
                cx, cy = self.on_center
                r = self.on_radius
                if self.power_on:
                    pygame.draw.circle(overlay, (60, 200, 90, 120), (cx, cy), r - 3)
                else:
                    pygame.draw.circle(overlay, (200, 60, 60, 90), (cx, cy), r - 3)
                pygame.draw.circle(overlay, (0, 0, 0, 140), (cx, cy), r, 2)

            # Knob: circular area and pointer line
            if self.knob_center and self.knob_radius:
                cx, cy = self.knob_center
                r = self.knob_radius
                pygame.draw.circle(overlay, (0, 0, 0, 150), (cx, cy), r, 2)
                # tick marks
                for i in range(-12, 13, 6):
                    a = (-150 + (i + 12) * (300 / 24.0)) * math.pi / 180.0
                    px = int(cx + (r - 4) * math.cos(a))
                    py = int(cy + (r - 4) * math.sin(a))
                    qx = int(cx + (r - 12) * math.cos(a))
                    qy = int(cy + (r - 12) * math.sin(a))
                    pygame.draw.line(overlay, (0, 0, 0, 160), (qx, qy), (px, py), 2)
                # Pointer based on current transpose
                a = (-150 + (self.transpose + 12) * (300 / 24.0)) * math.pi / 180.0
                px = int(cx + (r - 6) * math.cos(a))
                py = int(cy + (r - 6) * math.sin(a))
                pygame.draw.line(overlay, (10, 10, 10, 220), (cx, cy), (px, py), 4)

                # Label showing transpose value
                txt = f"Tune {self.transpose:+d}"
                img = self.font.render(txt, True, (0, 0, 0))
                overlay.blit(img, (cx - img.get_width() // 2, cy + r + 6))

            # Power status label
            lbl = "ON" if self.power_on else "OFF"
            col = (30, 150, 70) if self.power_on else (180, 50, 50)
            img = self.font.render(f"Power: {lbl}", True, col)
            overlay.blit(img, (int(self.w * 0.06), int(self.h * 0.08)))

            # Draw overlay onto screen
            screen.blit(overlay, (0, 0))

        # ------------- end of class -------------
    return PianoGame(screen, background_surf, context)
