import pygame
import numpy as np

# Module entry point
def create_game(screen, background_surf, context):
    """
    Returns an object with handle_event(event), update(dt), draw(screen).
    The image shows a simple piano with an "on" button. We overlay hitboxes
    proportionally to the background image and synthesize tones via context["audio"].
    """
    class PianoKey:
        def __init__(self, rect, midi, is_black):
            self.rect = pygame.Rect(rect)
            self.midi = midi
            self.is_black = is_black
            self.active = False
            self.last_play_time = 0.0

    class Game:
        def __init__(self, screen, background_surf, context):
            self.screen = screen
            self.bg = background_surf
            self.w, self.h = self.bg.get_size()

            # Audio helpers from context
            self.SAMPLE_RATE = context["audio"]["SAMPLE_RATE"]
            self.make_sound_from_wave = context["audio"]["make_sound_from_wave"]

            # Power state
            self.power_on = False

            # Simple state
            self.transpose = 0  # semitones
            self.master_volume = 0.8
            self.pressed_keys = set()
            self.time = 0.0

            # Layout (proportional to image)
            self.on_button_rect = self._make_on_button_rect()
            self.keybed_rect = self._make_keybed_rect()

            # Build keys (two octaves of white keys C..B, with appropriate black keys)
            self.white_keys, self.black_keys = self._build_piano_keys()

            # For drag-glissando dedup
            self._last_key_id = None

            # Font for small overlays
            pygame.font.init()
            self.font = pygame.font.SysFont(None, max(14, int(self.h * 0.03)))

        def _make_on_button_rect(self):
            # A rounded rectangle at top-right area in the image
            bw, bh = self.w, self.h
            btn_w = int(bw * 0.22)
            btn_h = int(bh * 0.13)
            x = int(bw * 0.62)
            y = int(bh * 0.15)
            return pygame.Rect(x, y, btn_w, btn_h)

        def _make_keybed_rect(self):
            # Large rounded rectangle with keys near the bottom half of the image
            bw, bh = self.w, self.h
            kb_w = int(bw * 0.72)
            kb_h = int(bh * 0.42)
            x = int(bw * 0.14)
            y = int(bh * 0.45)
            return pygame.Rect(x, y, kb_w, kb_h)

        def _build_piano_keys(self):
            # Two octaves of white keys: C..B (7 per octave) -> 14 white keys
            kb = self.keybed_rect
            n_white = 14
            gap = max(1, int(kb.w * 0.002))  # small gap so overlays don't fully cover art
            white_w = (kb.w - gap * (n_white - 1)) // n_white
            white_h = kb.h - int(kb.h * 0.06)  # leave a margin inside the drawn frame
            top = kb.y + (kb.h - white_h) // 2
            left = kb.x

            # MIDI mapping: base C4 = 60
            base_midi = 60  # C4
            white_scale = [0, 2, 4, 5, 7, 9, 11]  # offsets in an octave

            white_keys = []
            for i in range(n_white):
                octave = i // 7
                degree = i % 7
                midi = base_midi + white_scale[degree] + 12 * octave + self.transpose
                rect = (left + i * (white_w + gap), top, white_w, white_h)
                white_keys.append(PianoKey(rect, midi, is_black=False))

            # Black keys: pattern per octave: C#, D#, -, F#, G#, A#, -
            black_pattern_after_white_index = [0, 1, 3, 4, 5]  # indices within an octave that have black after them
            black_keys = []
            black_w = int(white_w * 0.6)
            black_h = int(white_h * 0.62)
            for i in range(n_white):
                idx_in_oct = i % 7
                octave = i // 7
                if idx_in_oct in black_pattern_after_white_index:
                    # Position black key centered at boundary between this white key and the next
                    left_i = left + i * (white_w + gap)
                    boundary_x = left_i + white_w + gap // 2
                    bx = int(boundary_x - black_w // 2)
                    by = top
                    midi = base_midi + (white_scale[idx_in_oct] + 1) + 12 * octave + self.transpose
                    black_keys.append(PianoKey((bx, by, black_w, black_h), midi, is_black=True))

            return white_keys, black_keys

        def _freq_from_midi(self, midi):
            return 440.0 * (2.0 ** ((midi - 69) / 12.0))

        def _play_tone(self, midi, velocity=1.0):
            if not self.power_on:
                return
            dur = 0.45 + 0.25 * float(np.clip(velocity, 0.0, 1.0))
            freq = self._freq_from_midi(midi + self.transpose)
            t = np.linspace(0, dur, int(self.SAMPLE_RATE * dur), endpoint=False)
            wave = 0.6 * np.sin(2 * np.pi * freq * t)

            # Short fade-in/out to avoid clicks
            fade_len = int(0.01 * self.SAMPLE_RATE)
            if fade_len > 0 and wave.size > 2 * fade_len:
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                wave[:fade_len] *= fade_in
                wave[-fade_len:] *= fade_out

            snd = self.make_sound_from_wave(wave, volume=self.master_volume)
            try:
                snd.play()
            except Exception:
                # If the host mixer isn't ready, silently ignore
                pass

        def _mouse_velocity_from_pos(self, y):
            # Higher velocity when clicking lower (closer to the bottom edge of the keybed)
            kb = self.keybed_rect
            rel = 1.0 - (y - kb.y) / max(1, kb.h)
            rel = np.clip(1.0 - rel, 0.0, 1.0)
            return float(rel)

        def _hit_test_key(self, pos):
            # Prioritize black keys (they sit on top visually)
            for i, k in enumerate(self.black_keys):
                if k.rect.collidepoint(pos):
                    return ('black', i, k)
            for i, k in enumerate(self.white_keys):
                if k.rect.collidepoint(pos):
                    return ('white', i, k)
            return (None, None, None)

        def _toggle_power(self):
            self.power_on = not self.power_on

        def handle_event(self, event):
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.on_button_rect.collidepoint(event.pos):
                    self._toggle_power()
                    self._last_key_id = None
                    return
                zone = self._hit_test_key(event.pos)
                if zone[0] is not None and self.power_on:
                    kind, idx, key = zone
                    key.active = True
                    self.pressed_keys.add((kind, idx))
                    vel = self._mouse_velocity_from_pos(event.pos[1])
                    self._play_tone(key.midi, velocity=vel)
                    self._last_key_id = (kind, idx)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                # Clear active states
                self.pressed_keys.clear()
                for k in self.white_keys + self.black_keys:
                    k.active = False
                self._last_key_id = None
            elif event.type == pygame.MOUSEMOTION:
                if event.buttons[0]:
                    # Dragging across keys (glissando)
                    zone = self._hit_test_key(event.pos)
                    if zone[0] is not None and self.power_on:
                        kind, idx, key = zone
                        if (kind, idx) != self._last_key_id:
                            # Update active flags
                            for k in self.white_keys + self.black_keys:
                                k.active = False
                            key.active = True
                            self.pressed_keys = {(kind, idx)}
                            vel = self._mouse_velocity_from_pos(event.pos[1])
                            self._play_tone(key.midi, velocity=vel)
                            self._last_key_id = (kind, idx)
            elif event.type == pygame.MOUSEWHEEL:
                # Use wheel to transpose in semitones (-12..+12)
                self.transpose = int(np.clip(self.transpose + event.y, -12, 12))
                # Rebuild key MIDI assignments while keeping geometry
                for i, k in enumerate(self.white_keys):
                    octave = i // 7
                    degree = i % 7
                    base_midi = 60  # C4
                    white_scale = [0, 2, 4, 5, 7, 9, 11]
                    k.midi = base_midi + white_scale[degree] + 12 * octave + self.transpose
                # Black keys
                for i, k in enumerate(self.black_keys):
                    # Recompute roughly based on position within octave
                    # Determine nearest white boundary index and resolve midi from x-position
                    # Simpler: infer from nearest white key center to the left and add +1 semitone
                    left_white = None
                    for w in self.white_keys:
                        if w.rect.centerx <= k.rect.centerx:
                            left_white = w
                        else:
                            break
                    if left_white is None:
                        left_white = self.white_keys[0]
                    k.midi = left_white.midi + 1

        def update(self, dt):
            self.time += dt

        def draw(self, screen):
            # Draw overlays only (background already blitted)
            # Power button highlight
            if self.power_on:
                col = (60, 220, 90, 140)
            else:
                col = (220, 60, 60, 120)
            s = pygame.Surface(self.on_button_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, col, s.get_rect(), border_radius=int(min(self.on_button_rect.w, self.on_button_rect.h) * 0.3))
            pygame.draw.rect(s, (0, 0, 0, 180), s.get_rect(), width=2, border_radius=int(min(self.on_button_rect.w, self.on_button_rect.h) * 0.3))
            screen.blit(s, self.on_button_rect.topleft)

            # Small status text
            txt = f"{'ON' if self.power_on else 'OFF'}  transpose: {self.transpose:+d}"
            text_surf = self.font.render(txt, True, (0, 0, 0))
            screen.blit(text_surf, (self.on_button_rect.x, self.on_button_rect.bottom + int(self.h * 0.01)))

            # Dim keyboard if power is off
            if not self.power_on:
                dim = pygame.Surface(self.keybed_rect.size, pygame.SRCALPHA)
                dim.fill((0, 0, 0, 80))
                screen.blit(dim, self.keybed_rect.topleft)

            # Key highlights (translucent so original art remains visible)
            # White key highlights
            for k in self.white_keys:
                if k.active:
                    c = (120, 180, 255, 90)
                else:
                    c = (200, 220, 255, 40)
                s = pygame.Surface(k.rect.size, pygame.SRCALPHA)
                pygame.draw.rect(s, c, s.get_rect())
                screen.blit(s, k.rect.topleft)

            # Black key highlights on top
            for k in self.black_keys:
                if k.active:
                    c = (255, 210, 80, 120)
                else:
                    c = (0, 0, 0, 0)  # don't overlay idle black keys to keep the image clean
                if c[-1] > 0:
                    s = pygame.Surface(k.rect.size, pygame.SRCALPHA)
                    pygame.draw.rect(s, c, s.get_rect())
                    screen.blit(s, k.rect.topleft)

            # Subtle outline of keybed hit area (to indicate interactivity)
            pygame.draw.rect(screen, (0, 0, 0), self.keybed_rect, 2, border_radius=int(min(self.keybed_rect.w, self.keybed_rect.h) * 0.06))

    return Game(screen, background_surf, context)
