import pygame
import numpy as np

def create_game(screen, background_surf, context):
    class SimpleTimer:
        def __init__(self, screen, bg, context):
            self.screen = screen
            self.bg = bg
            self.w, self.h = bg.get_size()
            self.font = pygame.font.SysFont(None, max(20, self.h // 12))
            self.small_font = pygame.font.SysFont(None, max(18, self.h // 20))
            self.running = False
            self.elapsed = 0.0
            self.hover_idx = -1
            self.last_mouse_pos = (0, 0)

            # Audio helpers from context
            audio = context.get("audio", {})
            self.SR = int(audio.get("SAMPLE_RATE", 44100))
            self.make_sound_from_wave = audio.get("make_sound_from_wave", None)

            # Detect interactive regions from the image
            self.button_defs, self.display_rect = self._detect_regions()

            # If detection fails, define reasonable fallbacks
            if not self.button_defs:
                self.button_defs = self._fallback_buttons()
            if self.display_rect is None:
                self.display_rect = self._fallback_display()

        # ---------- Detection ----------
        def _detect_regions(self):
            try:
                arr = pygame.surfarray.array3d(self.bg)  # shape (w, h, 3)
            except Exception:
                return [], None

            # Convert to (h, w, 3)
            arr = np.transpose(arr, (1, 0, 2)).astype(np.float32)
            h, w, _ = arr.shape
            # Luminance and "darkness"
            gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            dark = 255.0 - gray

            # Vertical projection over middle columns
            x0 = int(0.2 * w)
            x1 = int(0.8 * w)
            vert_profile = dark[:, x0:x1].sum(axis=1)

            # Smooth profile
            k = max(7, int(h * 0.015) | 1)  # odd
            kernel = np.ones(k, dtype=np.float32) / k
            vert_smooth = np.convolve(vert_profile, kernel, mode="same")

            # Peak picking
            max_v = vert_smooth.max() if vert_smooth.size else 0
            if max_v <= 0:
                return [], None
            thresh = max_v * 0.55
            min_gap = max(15, int(h * 0.04))

            peaks = []
            for y in range(1, h - 1):
                v = vert_smooth[y]
                if v > thresh and v >= vert_smooth[y - 1] and v >= vert_smooth[y + 1]:
                    if not peaks or (y - peaks[-1] > min_gap):
                        peaks.append(y)
                    else:
                        # Keep the higher peak if too close
                        if v > vert_smooth[peaks[-1]]:
                            peaks[-1] = y

            peaks = sorted(peaks)
            # We expect around 4 text lines: Timer, Start, Stop, Reset
            # Build bounding boxes around each peak by horizontal projection
            def box_for_peak(yc):
                half = max(6, int(h * 0.03))
                y0 = max(0, yc - half)
                y1 = min(h, yc + half)
                row_band = dark[y0:y1, :]
                horiz = row_band.sum(axis=0)
                # Smooth horizontally
                hk = max(7, int(w * 0.01) | 1)
                hker = np.ones(hk, dtype=np.float32) / hk
                horiz = np.convolve(horiz, hker, mode="same")
                if horiz.size == 0:
                    return None
                ht = horiz.max() * 0.35
                above = (horiz > ht).astype(np.int8)
                # Find widest contiguous segment
                best_len, best_seg = 0, None
                in_seg = False
                s = 0
                for i, val in enumerate(above):
                    if val and not in_seg:
                        in_seg = True
                        s = i
                    elif not val and in_seg:
                        in_seg = False
                        e = i
                        if (e - s) > best_len:
                            best_len = e - s
                            best_seg = (s, e)
                if in_seg:
                    e = len(above)
                    if (e - s) > best_len:
                        best_len = e - s
                        best_seg = (s, e)
                if not best_seg:
                    return None
                pad_x = max(6, int(w * 0.01))
                x0b = max(0, best_seg[0] - pad_x)
                x1b = min(w, best_seg[1] + pad_x)
                pad_y = max(6, int(h * 0.01))
                y0b = max(0, y0 - pad_y)
                y1b = min(h, y1 + pad_y)
                return pygame.Rect(x0b, y0b, x1b - x0b, y1b - y0b)

            boxes = []
            for p in peaks:
                rect = box_for_peak(p)
                if rect and rect.w > w * 0.08 and rect.h > h * 0.02:
                    boxes.append((p, rect))

            boxes.sort(key=lambda t: t[0])

            # Classify: Skip first "Timer" if we have >=4, then next three are Start/Stop/Reset
            button_rects = []
            text_lines = [b[1] for b in boxes]
            if len(text_lines) >= 4:
                candidates = text_lines[1:4]
            elif len(text_lines) == 3:
                candidates = text_lines  # assume they are Start/Stop/Reset
            else:
                candidates = []

            labels = ["Start", "Stop", "Reset"]
            for i, r in enumerate(candidates[:3]):
                # Slightly expand to make clicking easier
                rr = r.inflate(int(0.05 * r.w) + 10, int(0.25 * r.h) + 10)
                button_rects.append({"name": labels[i], "rect": rr})

            # Display area: aim between first and second text lines (Timer and Start)
            display_rect = None
            if len(boxes) >= 2:
                top = boxes[0][1]
                nxt = boxes[1][1]
                cy = (top.bottom + nxt.top) // 2
                disp_h = max(24, int(0.12 * h))
                disp_w = max(120, int(0.6 * w))
                cx = w // 2
                display_rect = pygame.Rect(0, 0, disp_w, disp_h)
                display_rect.center = (cx, cy)
            return button_rects, display_rect

        def _fallback_buttons(self):
            # Place three stacked buttons roughly in lower middle
            cx = self.w // 2
            bw = max(160, int(self.w * 0.36))
            bh = max(38, int(self.h * 0.07))
            gap = max(10, int(self.h * 0.02))
            start_y = int(self.h * 0.50)
            rects = []
            for i, name in enumerate(["Start", "Stop", "Reset"]):
                r = pygame.Rect(0, 0, bw, bh)
                r.center = (cx, start_y + i * (bh + gap))
                rects.append({"name": name, "rect": r})
            return rects

        def _fallback_display(self):
            disp_w = max(180, int(self.w * 0.6))
            disp_h = max(40, int(self.h * 0.12))
            r = pygame.Rect(0, 0, disp_w, disp_h)
            r.center = (self.w // 2, int(self.h * 0.33))
            return r

        # ---------- Audio ----------
        def _play_tone(self, freq=440.0, dur=0.12, volume=0.8):
            if not self.make_sound_from_wave:
                return
            sr = self.SR
            dur = max(0.02, float(dur))
            t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
            wave = 0.6 * np.sin(2 * np.pi * float(freq) * t).astype(np.float32)
            # Quick fade in/out to avoid clicks
            f_samp = max(1, int(0.01 * sr))  # 10ms
            env = np.ones_like(wave)
            fade = np.linspace(0.0, 1.0, f_samp, dtype=np.float32)
            env[:f_samp] *= fade
            env[-f_samp:] *= fade[::-1]
            wave *= env
            snd = self.make_sound_from_wave(wave, volume=volume)
            try:
                snd.play()
            except Exception:
                pass

        def _play_reset_chirp(self):
            # Two quick tones to indicate reset
            self._play_tone(540, 0.07, 0.8)
            # Schedule a follow-up tone using a lightweight timer via pygame events
            pygame.time.set_timer(pygame.USEREVENT + 1, 90, True)

        # ---------- Event handling ----------
        def handle_event(self, event):
            if event.type == pygame.MOUSEMOTION:
                self.last_mouse_pos = event.pos
                self.hover_idx = -1
                for i, b in enumerate(self.button_defs):
                    if b["rect"].collidepoint(event.pos):
                        self.hover_idx = i
                        break
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                for i, b in enumerate(self.button_defs):
                    if b["rect"].collidepoint(pos):
                        name = b["name"]
                        if name == "Start":
                            if not self.running:
                                self._play_tone(880, 0.10, 0.8)
                            self.running = True
                        elif name == "Stop":
                            if self.running:
                                self._play_tone(660, 0.10, 0.8)
                            self.running = False
                        elif name == "Reset":
                            self.elapsed = 0.0
                            self.running = False
                            self._play_reset_chirp()
                        break
            elif event.type == pygame.USEREVENT + 1:
                # Second tone of reset chirp
                self._play_tone(360, 0.08, 0.8)

        # ---------- Update/Draw ----------
        def update(self, dt):
            # Try to handle dt in ms if needed
            if dt is None:
                dt = 0.0
            if dt > 2.0:  # likely milliseconds
                dt = dt / 1000.0
            if self.running:
                self.elapsed = max(0.0, self.elapsed + float(dt))

        def _format_time(self, t):
            # mm:ss
            total = int(t + 0.5)  # round to nearest second for display
            m = total // 60
            s = total % 60
            return f"{m:02d}:{s:02d}"

        def draw(self, screen):
            # Draw semi-transparent overlay for display area
            disp = self.display_rect
            # Slight insets for better aesthetics
            disp_draw = disp.inflate(0, 0)
            # Draw display background
            surf = pygame.Surface(disp_draw.size, pygame.SRCALPHA)
            bg_col = (255, 255, 255, 220)  # slightly translucent white
            pygame.draw.rect(surf, bg_col, surf.get_rect(), border_radius=min(20, disp_draw.h // 3))
            pygame.draw.rect(surf, (40, 40, 40), surf.get_rect(), width=3, border_radius=min(20, disp_draw.h // 3))
            screen.blit(surf, disp_draw.topleft)

            # Render time text
            txt = self._format_time(self.elapsed)
            color = (40, 40, 40)
            if self.running:
                color = (0, 128, 0)
            text_surf = self.font.render(txt, True, color)
            text_rect = text_surf.get_rect(center=disp_draw.center)
            screen.blit(text_surf, text_rect)

            # Draw buttons as transparent highlights over the labeled words
            for i, b in enumerate(self.button_defs):
                r = b["rect"]
                hover = (i == self.hover_idx)
                base_color = (0, 120, 255, 70) if b["name"] == "Start" else ((255, 120, 0, 70) if b["name"] == "Stop" else (120, 120, 120, 60))
                border_color = (0, 120, 255) if b["name"] == "Start" else ((200, 60, 0) if b["name"] == "Stop" else (80, 80, 80))
                if hover:
                    base_color = (min(base_color[0] + 40, 255), min(base_color[1] + 40, 255), min(base_color[2] + 40, 255), 110)
                hs = pygame.Surface(r.size, pygame.SRCALPHA)
                pygame.draw.rect(hs, base_color, hs.get_rect(), border_radius=min(14, r.h // 3))
                pygame.draw.rect(hs, border_color, hs.get_rect(), width=2, border_radius=min(14, r.h // 3))
                screen.blit(hs, r.topleft)

                # Optional small overlay label to reinforce click targets
                label = self.small_font.render(b["name"], True, border_color)
                lr = label.get_rect(center=r.center)
                screen.blit(label, lr)

    return SimpleTimer(screen, background_surf, context)
