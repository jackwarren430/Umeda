import pygame as pg
import numpy as np

def create_game(screen, background_surf, context):
    class PianoGame:
        def __init__(self, screen, context):
            self.screen = screen
            self.rect = screen.get_rect()
            self.font = pg.font.SysFont(None, max(16, self.rect.w // 40))

            # Audio helpers (provided by shell via context)
            self.SAMPLE_RATE = context["audio"]["SAMPLE_RATE"]
            self.make_sound_from_wave = context["audio"]["make_sound_from_wave"]

            # UI state
            self.power_on = False
            self.tune = 0.0  # semitones, controlled by slider (-12 .. +12)
            self.dragging_slider = False
            self.last_key_id = None
            self.pressed_keys = set()

            # Layout and keys
            self._layout()
            self._build_keys()

            # Cache generated sounds by (midi, tune_round, dur)
            self.sound_cache = {}

        # ---------- Layout ----------
        def _layout(self):
            r = self.rect
            pad_w = int(r.w * 0.08)
            pad_h = int(r.h * 0.12)
            self.panel = pg.Rect(r.x + pad_w, r.y + pad_h, r.w - 2 * pad_w, r.h - 2 * pad_h)

            # Keyboard area on the left ~70% width, bottom 45% height of panel
            kb_margin = int(min(self.panel.w, self.panel.h) * 0.06)
            kb_w = int(self.panel.w * 0.68)
            kb_h = int(self.panel.h * 0.38)
            kb_x = self.panel.x + kb_margin
            kb_y = self.panel.bottom - kb_margin - kb_h
            self.kb_rect = pg.Rect(kb_x, kb_y, kb_w, kb_h)

            # On/Off button near top-right area of panel
            btn_w = int(self.panel.w * 0.18)
            btn_h = int(self.panel.h * 0.16)
            btn_x = self.panel.right - btn_w - int(self.panel.w * 0.24)
            btn_y = self.panel.y + int(self.panel.h * 0.18)
            self.on_btn = pg.Rect(btn_x, btn_y, btn_w, btn_h)

            # Slider (vertical) on the far right side of panel
            track_h = int(self.panel.h * 0.64)
            track_w = int(max(6, self.panel.w * 0.012))
            track_x = self.panel.right - int(self.panel.w * 0.09)
            track_y = self.panel.y + (self.panel.h - track_h) // 2
            self.slider_track = pg.Rect(track_x, track_y, track_w, track_h)
            knob_w = int(track_w * 3.3)
            knob_h = int(self.panel.h * 0.07)
            # knob y determined by tune value (map in set/get functions)
            self.knob_rect = pg.Rect(0, 0, knob_w, knob_h)
            self._update_knob_from_tune()

        def _update_knob_from_tune(self):
            # Map tune (-12..+12) to knob y within track
            tmin, tmax = -12.0, 12.0
            frac = (self.tune - tmin) / (tmax - tmin)
            frac = max(0.0, min(1.0, frac))
            y = self.slider_track.y + int(frac * (self.slider_track.h - self.knob_rect.h))
            self.knob_rect.centerx = self.slider_track.centerx
            self.knob_rect.y = y

        def _update_tune_from_knob(self):
            # Inverse of above
            tmin, tmax = -12.0, 12.0
            span = self.slider_track.h - self.knob_rect.h
            if span <= 0:
                self.tune = 0.0
                return
            frac = (self.knob_rect.y - self.slider_track.y) / span
            frac = max(0.0, min(1.0, frac))
            self.tune = tmin + frac * (tmax - tmin)

        # ---------- Keys ----------
        def _build_keys(self):
            # Build a single octave: C4..B4 (white keys) + sharps (black)
            kb = self.kb_rect
            n_white = 7
            white_w = kb.w / n_white

            self.white_keys = []
            self.black_keys = []

            white_midis = [60, 62, 64, 65, 67, 69, 71]  # C D E F G A B (octave 4)
            # Positions that have black keys between white keys (after indices 0,1,3,4,5)
            black_after_white_idx = [0, 1, 3, 4, 5]
            black_midis = [61, 63, 66, 68, 70]  # C# D# F# G# A#

            # Create white key rects
            for i in range(n_white):
                x = kb.x + int(i * white_w)
                rect = pg.Rect(x, kb.y, int(white_w) - 1, kb.h)
                self.white_keys.append({"rect": rect, "midi": white_midis[i], "id": f"W{i}"})

            # Create black key rects
            b_w = int(white_w * 0.6)
            b_h = int(kb.h * 0.6)
            for j, wi in enumerate(black_after_white_idx):
                # place centered between white key wi and wi+1, slightly shifted right
                x_left_white = kb.x + int((wi + 1) * white_w)
                x = x_left_white - b_w // 2
                rect = pg.Rect(x, kb.y, b_w, b_h)
                self.black_keys.append({"rect": rect, "midi": black_midis[j], "id": f"B{j}"})

        # ---------- Audio ----------
        def midi_to_freq(self, m):
            return 440.0 * (2.0 ** ((m - 69) / 12.0))

        def play_note(self, midi, dur=0.6):
            # Apply tuning in semitones via slider
            freq = self.midi_to_freq(midi) * (2.0 ** (self.tune / 12.0))
            t = np.linspace(0, dur, int(self.SAMPLE_RATE * dur), endpoint=False)
            wave = 0.6 * np.sin(2 * np.pi * freq * t)  # add short fade in/out to avoid clicks
            # Short fade (5 ms) in/out
            fade_n = max(1, int(0.005 * self.SAMPLE_RATE))
            if fade_n * 2 < wave.size:
                fade_in = np.linspace(0.0, 1.0, fade_n)
                fade_out = fade_in[::-1]
                wave[:fade_n] *= fade_in
                wave[-fade_n:] *= fade_out
            snd = self.make_sound_from_wave(wave, volume=0.8)
            snd.play()

        # ---------- Event handling ----------
        def _key_at_pos(self, pos):
            # Check black keys first (they sit on top)
            for k in self.black_keys:
                if k["rect"].collidepoint(pos):
                    return k
            for k in self.white_keys:
                if k["rect"].collidepoint(pos):
                    return k
            return None

        def handle_event(self, event):
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                # On/Off toggle
                if self.on_btn.collidepoint(pos):
                    self.power_on = not self.power_on
                    return

                # Slider drag check
                if self.knob_rect.inflate(12, 12).collidepoint(pos) or self.slider_track.inflate(10, 10).collidepoint(pos):
                    self.dragging_slider = True
                    # Snap knob to click position
                    self.knob_rect.centery = pos[1]
                    # Clamp within track
                    if self.knob_rect.top < self.slider_track.top:
                        self.knob_rect.top = self.slider_track.top
                    if self.knob_rect.bottom > self.slider_track.bottom:
                        self.knob_rect.bottom = self.slider_track.bottom
                    self._update_tune_from_knob()
                    return

                # Keys
                k = self._key_at_pos(pos)
                if k is not None:
                    self.last_key_id = k["id"]
                    if self.power_on:
                        self.play_note(k["midi"])
                    self.pressed_keys.add(k["id"])

            elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
                self.dragging_slider = False
                self.last_key_id = None
                self.pressed_keys.clear()

            elif event.type == pg.MOUSEMOTION:
                if self.dragging_slider:
                    # Drag knob
                    self.knob_rect.centery = event.pos[1]
                    if self.knob_rect.top < self.slider_track.top:
                        self.knob_rect.top = self.slider_track.top
                    if self.knob_rect.bottom > self.slider_track.bottom:
                        self.knob_rect.bottom = self.slider_track.bottom
                    self._update_tune_from_knob()
                else:
                    # Glide over keys while holding mouse
                    buttons = pg.mouse.get_pressed(3)
                    if buttons[0]:
                        k = self._key_at_pos(event.pos)
                        if k is not None and k["id"] != self.last_key_id:
                            self.last_key_id = k["id"]
                            if self.power_on:
                                self.play_note(k["midi"])
                            self.pressed_keys = {k["id"]}

        # ---------- Update/Draw ----------
        def update(self, dt):
            pass

        def draw(self, screen):
            # Background already drawn by shell; draw overlays only
            # Draw keyboard overlays (outlines + pressed highlights)
            # Semi-transparent layer for highlights
            # White keys
            for k in self.white_keys:
                rect = k["rect"]
                if k["id"] in self.pressed_keys:
                    s = pg.Surface(rect.size, pg.SRCALPHA)
                    s.fill((255, 230, 120, 110) if self.power_on else (160, 160, 160, 90))
                    screen.blit(s, rect.topleft)
                pg.draw.rect(screen, (20, 20, 20), rect, 2, border_radius=6)

            # Black keys
            for k in self.black_keys:
                rect = k["rect"]
                if k["id"] in self.pressed_keys:
                    s = pg.Surface(rect.size, pg.SRCALPHA)
                    s.fill((120, 220, 255, 120) if self.power_on else (120, 120, 120, 90))
                    screen.blit(s, rect.topleft)
                pg.draw.rect(screen, (20, 20, 20), rect, 0, border_radius=4)  # solid to match black keys
                pg.draw.rect(screen, (240, 240, 240), rect, 2, border_radius=4)

            # If power is off, dim keyboard area slightly
            if not self.power_on:
                dim = pg.Surface(self.kb_rect.size, pg.SRCALPHA)
                dim.fill((50, 50, 50, 60))
                screen.blit(dim, self.kb_rect.topleft)

            # On/Off button
            btn_color = (70, 190, 90) if self.power_on else (180, 180, 180)
            pg.draw.rect(screen, btn_color, self.on_btn, border_radius=16)
            pg.draw.rect(screen, (30, 30, 30), self.on_btn, 3, border_radius=16)
            label = "on" if self.power_on else "off"
            txt = self.font.render(label, True, (20, 20, 20))
            screen.blit(txt, txt.get_rect(center=self.on_btn.center))

            # Slider track
            # Track line
            pg.draw.rect(screen, (20, 20, 20), self.slider_track, border_radius=6)
            inner = self.slider_track.inflate(-self.slider_track.w + 4, -4)
            pg.draw.rect(screen, (230, 230, 230), inner, border_radius=6)
            # Knob
            pg.draw.rect(screen, (30, 30, 30), self.knob_rect, border_radius=12)
            pg.draw.rect(screen, (230, 230, 230), self.knob_rect.inflate(-6, -6), 0, border_radius=10)

            # Slider ticks and value
            tmin, tmax = -12, 12
            for i in range(5):
                frac = i / 4.0
                y = int(self.slider_track.y + frac * (self.slider_track.h - 1))
                pg.draw.line(screen, (60, 60, 60), (self.slider_track.right + 6, y), (self.slider_track.right + 18, y), 2)
            tune_text = f"tune {self.tune:+.1f} st"
            timg = self.font.render(tune_text, True, (20, 20, 20))
            trect = timg.get_rect()
            trect.midtop = (self.slider_track.centerx, self.slider_track.bottom + 6)
            screen.blit(timg, trect)

    return PianoGame(screen, context)
