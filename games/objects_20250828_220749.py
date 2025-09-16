import pygame
import numpy as np

# This module builds an interactive piano overlay with an ON button and a tuning slider.
# It assumes the provided background already shows a piano, button, and slider.
# We place interactive hitboxes where those elements are likely located.

def create_game(screen, background_surf, context):
    class PianoGame:
        def __init__(self, screen, background_surf, context):
            self.screen = screen
            self.bg = background_surf
            self.w, self.h = self.screen.get_size()
            self.font = pygame.font.SysFont(None, max(18, int(self.h * 0.05)))
            self.small_font = pygame.font.SysFont(None, max(14, int(self.h * 0.035)))

            # Audio helpers from context
            self.SAMPLE_RATE = context["audio"]["SAMPLE_RATE"]
            self.make_sound_from_wave = context["audio"]["make_sound_from_wave"]

            # State
            self.power_on = False
            self.dragging_slider = False
            self.slider_value = 0.0  # semitone transpose in [-12, 12]
            self.slider_min = -12.0
            self.slider_max = 12.0
            self.mouse_down = False
            self.current_key_id = None  # track for glissando without spamming

            # Geometry (ratios chosen to match the provided drawing)
            self._compute_layout()

            # Build key hitboxes and note mapping
            self._build_keys()

        def _compute_layout(self):
            w, h = self.w, self.h
            # Keyboard box roughly centered lower-half
            kb_left = int(w * 0.17)
            kb_right = int(w * 0.83)
            kb_top = int(h * 0.42)
            kb_bottom = int(h * 0.76)
            self.keyboard_rect = pygame.Rect(kb_left, kb_top, kb_right - kb_left, kb_bottom - kb_top)

            # ON button near top-right of the keyboard area (matching the drawing)
            on_w = int(w * 0.16)
            on_h = int(h * 0.10)
            on_x = int(w * 0.58)
            on_y = int(h * 0.16)
            self.on_rect = pygame.Rect(on_x, on_y, on_w, on_h)

            # Slider track on far right
            track_x = int(w * 0.89)
            track_y1 = int(h * 0.18)
            track_y2 = int(h * 0.80)
            self.slider_track_rect = pygame.Rect(track_x - int(w * 0.006), track_y1, int(w * 0.012), track_y2 - track_y1)

        def _build_keys(self):
            # Build a 14-white-key section (about two octaves) starting at MIDI 60 (C4)
            self.white_key_count = 14
            self.white_rects = []
            self.white_midis = []
            self.black_rects = []
            self.black_midis = []

            kb = self.keyboard_rect
            wkw = kb.width / self.white_key_count
            wkh = kb.height

            # semitone offsets for white keys within an octave starting at C
            white_offsets = [0, 2, 4, 5, 7, 9, 11]  # C D E F G A B
            # black keys occur after white indices (in an octave) 0(C),1(D),3(F),4(G),5(A)
            black_after_white_idx = {0: 1, 1: 3, 3: 6, 4: 8, 5: 10}  # maps local white index -> semitone offset within octave

            start_midi = 60  # C4
            for i in range(self.white_key_count):
                x = int(kb.left + i * wkw)
                rect = pygame.Rect(x, kb.top, int(np.ceil(wkw)), int(wkh))
                self.white_rects.append(rect)

                octave = i // 7
                in_oct_index = i % 7
                semitone = white_offsets[in_oct_index] + 12 * octave
                self.white_midis.append(start_midi + semitone)

            # Black keys
            black_w = wkw * 0.6
            black_h = wkh * 0.62
            for i in range(self.white_key_count):
                local = i % 7
                if local in black_after_white_idx and i < self.white_key_count - 1:
                    # place black key centered between white i and i+1
                    left_white = self.white_rects[i]
                    # position black so it's roughly overlapping the seam
                    bx = left_white.right - int(black_w // 2)
                    by = self.keyboard_rect.top
                    brect = pygame.Rect(int(bx), int(by), int(black_w), int(black_h))
                    self.black_rects.append(brect)

                    octave = i // 7
                    semitone = black_after_white_idx[local] + 12 * octave
                    self.black_midis.append(start_midi + semitone)

        def _midi_to_freq(self, midi):
            return 440.0 * (2.0 ** ((midi - 69) / 12.0))

        def _play_note(self, midi, dur=0.6):
            if not self.power_on:
                return
            freq = self._midi_to_freq(midi + self.slider_value)
            # AUDIO CONTRACT
            t = np.linspace(0, dur, int(self.SAMPLE_RATE * dur), endpoint=False)
            wave = 0.6 * np.sin(2 * np.pi * freq * t)  # add short fade in/out to avoid clicks
            # small fade-in/out
            fade = int(max(1, self.SAMPLE_RATE * 0.01))
            if fade * 2 < len(wave):
                env = np.ones_like(wave)
                env[:fade] = np.linspace(0, 1, fade, endpoint=False)
                env[-fade:] = np.linspace(1, 0, fade, endpoint=False)
                wave = wave * env
            snd = self.make_sound_from_wave(wave, volume=0.8)
            snd.play()

        def _pos_to_key(self, pos):
            # Returns (is_black, index, midi) or None
            x, y = pos
            if not self.keyboard_rect.collidepoint(pos):
                return None
            # Check black keys first (they sit on top)
            for idx, r in enumerate(self.black_rects):
                if r.collidepoint(pos):
                    return True, idx, self.black_midis[idx]
            # Then white keys
            for idx, r in enumerate(self.white_rects):
                if r.collidepoint(pos):
                    return False, idx, self.white_midis[idx]
            return None

        def _update_slider_from_pos(self, y):
            # Map y in track to slider_value in [min, max]
            tr = self.slider_track_rect
            y = np.clip(y, tr.top, tr.bottom)
            t = 1.0 - (y - tr.top) / tr.height  # 1 at top, 0 at bottom
            val = self.slider_min + t * (self.slider_max - self.slider_min)
            # Quantize to 0.1 semitones for smoothness but readable changes
            self.slider_value = float(np.round(val * 10) / 10.0)

        def handle_event(self, event):
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.mouse_down = True
                if self.on_rect.collidepoint(event.pos):
                    self.power_on = not self.power_on
                    # Tiny confirmation blip when turning on
                    if self.power_on:
                        self._play_note(84, dur=0.15)  # C6 blip
                    else:
                        # no sound when turning off
                        pass
                elif self.slider_track_rect.inflate(12, 12).collidepoint(event.pos):
                    self.dragging_slider = True
                    self._update_slider_from_pos(event.pos[1])
                else:
                    hit = self._pos_to_key(event.pos)
                    if hit is not None:
                        is_black, idx, midi = hit
                        self.current_key_id = ("b" if is_black else "w", idx)
                        self._play_note(midi)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.mouse_down = False
                self.dragging_slider = False
                self.current_key_id = None

            elif event.type == pygame.MOUSEMOTION:
                if self.dragging_slider:
                    self._update_slider_from_pos(event.pos[1])
                elif self.mouse_down:
                    hit = self._pos_to_key(event.pos)
                    if hit is not None:
                        is_black, idx, midi = hit
                        key_id = ("b" if is_black else "w", idx)
                        if key_id != self.current_key_id:
                            self.current_key_id = key_id
                            self._play_note(midi)

        def update(self, dt):
            pass  # No time-based animation required

        def _draw_button(self):
            r = self.on_rect
            # Button background
            col = (30, 200, 90) if self.power_on else (180, 180, 180)
            pygame.draw.rect(self.screen, col, r, border_radius=int(r.height * 0.25))
            pygame.draw.rect(self.screen, (0, 0, 0), r, width=3, border_radius=int(r.height * 0.25))
            label = "ON" if self.power_on else "off"
            text = self.font.render(label, True, (0, 0, 0))
            self.screen.blit(text, text.get_rect(center=r.center))

        def _draw_slider(self):
            tr = self.slider_track_rect
            # Track
            pygame.draw.rect(self.screen, (0, 0, 0), tr, width=3, border_radius=6)
            # Knob position from value
            t = (self.slider_value - self.slider_min) / (self.slider_max - self.slider_min)
            y = int(tr.bottom - t * tr.height)
            # Knob - a horizontal bar
            knob_w = int(self.w * 0.045)
            knob_h = int(self.h * 0.012)
            knob_rect = pygame.Rect(tr.centerx - knob_w // 2, y - knob_h // 2, knob_w, knob_h)
            pygame.draw.rect(self.screen, (30, 30, 30), knob_rect, border_radius=4)
            pygame.draw.rect(self.screen, (230, 230, 230), knob_rect.inflate(-6, -6), border_radius=4)

            # Scale markers
            for s in [-12, -6, 0, 6, 12]:
                tt = (s - self.slider_min) / (self.slider_max - self.slider_min)
                yy = int(tr.bottom - tt * tr.height)
                pygame.draw.line(self.screen, (0, 0, 0), (tr.left - 10, yy), (tr.left - 2, yy), 2)
                label = self.small_font.render(str(s), True, (0, 0, 0))
                self.screen.blit(label, (tr.left - 40, yy - label.get_height() // 2))

            # Current value
            vlabel = self.small_font.render(f"Tune: {self.slider_value:+.1f} st", True, (0, 0, 0))
            self.screen.blit(vlabel, (tr.left - int(self.w * 0.12), tr.top - int(self.h * 0.04)))

        def _draw_key_overlays(self):
            # Visual feedback when pressing keys: subtle highlight
            if self.current_key_id is None:
                return
            kind, idx = self.current_key_id
            color = (0, 200, 255, 120) if kind == "w" else (255, 80, 80, 180)
            surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            rect = self.white_rects[idx] if kind == "w" else self.black_rects[idx]
            pygame.draw.rect(surf, color, rect)
            self.screen.blit(surf, (0, 0))

            # Outline whole keyboard area to communicate interactivity
            pygame.draw.rect(self.screen, (0, 0, 0), self.keyboard_rect, 2, border_radius=8)

        def draw(self, screen):
            # background_surf is already blitted by the shell; draw overlays only
            self._draw_button()
            self._draw_slider()
            self._draw_key_overlays()

    return PianoGame(screen, background_surf, context)
