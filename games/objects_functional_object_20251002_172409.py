import pygame
import numpy as np

def create_game(screen, background_surf, context):
    """
    Returns an object with methods:
      handle_event(event)
      update(dt)
      draw(screen)
    """
    SAMPLE_RATE = context["audio"]["SAMPLE_RATE"]
    make_sound_from_wave = context["audio"]["make_sound_from_wave"]

    class PianoGame:
        def __init__(self):
            self.screen_w, self.screen_h = screen.get_size()

            # Optional decorative sprite (piano drawing)
            self.sprite = context.get("sprite", None)
            if self.sprite and "rect" in self.sprite:
                self.sprite_rect = self.sprite["rect"].copy()
                sx, sy, sw, sh = self.sprite_rect
                # Layout inside sprite bounds
                self.keys_frame = pygame.Rect(
                    int(sx + 0.13 * sw),
                    int(sy + 0.50 * sh),
                    int(0.74 * sw),
                    int(0.32 * sh),
                )
                self.power_rect = pygame.Rect(
                    int(sx + 0.63 * sw),
                    int(sy + 0.13 * sh),
                    int(0.20 * sw),
                    int(0.12 * sh),
                )
            else:
                # Fallback layout if no sprite present
                pad = 20
                fw = int(self.screen_w * 0.72)
                fh = int(self.screen_h * 0.28)
                fx = (self.screen_w - fw) // 2
                fy = int(self.screen_h * 0.56)
                self.keys_frame = pygame.Rect(fx, fy, fw, fh)
                self.power_rect = pygame.Rect(
                    int(self.screen_w * 0.72),
                    int(self.screen_h * 0.10),
                    int(self.screen_w * 0.18),
                    int(self.screen_h * 0.10),
                )

            # Build keys (one octave: C D E F G A B + sharps)
            self.white_names = ["C", "D", "E", "F", "G", "A", "B"]
            self.black_map = {0: "C#", 1: "D#", 3: "F#", 4: "G#", 5: "A#"}  # index in whites
            self._build_key_rects()

            # Precompute sounds
            self.duration = 0.5
            self.sounds = {}
            for name, freq in self._note_freqs().items():
                self.sounds[name] = self._make_tone(freq, self.duration)

            # Interaction state
            self.power_on = False
            self.hover_power = False
            self.hover_key = None
            self.active_times = {k["name"]: 0.0 for k in self.white_keys + self.black_keys}

            # HUD font
            self.font_small = pygame.font.SysFont(None, 18)
            self.font_med = pygame.font.SysFont(None, 24)

        def _build_key_rects(self):
            # White keys
            kf = self.keys_frame
            w = kf.width / 7.0
            self.white_keys = []
            for i, name in enumerate(self.white_names):
                rect = pygame.Rect(int(kf.x + i * w), kf.y, int(round(w)), kf.height)
                self.white_keys.append({"name": name, "rect": rect, "black": False})
            # Black keys
            self.black_keys = []
            black_w = int(max(6, w * 0.6))
            black_h = int(self.keys_frame.height * 0.6)
            for i, name in self.black_map.items():
                center_x = self.white_keys[i]["rect"].right
                rect = pygame.Rect(0, 0, black_w, black_h)
                rect.centerx = center_x
                rect.top = self.keys_frame.top
                # Clamp within frame
                rect.left = max(rect.left, self.keys_frame.left)
                rect.right = min(rect.right, self.keys_frame.right)
                self.black_keys.append({"name": name, "rect": rect, "black": True})

        def _note_freqs(self):
            # Reference A4 = 440 Hz; octave 4 for C..B
            # Semitone offsets relative to A4
            offsets = {
                "C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5,
                "F": -4, "F#": -3, "G": -2, "G#": -1, "A": 0,
                "A#": 1, "B": 2
            }
            notes = {}
            for n in ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]:
                notes[n] = 440.0 * (2 ** (offsets[n] / 12.0))
            return notes

        def _make_tone(self, freq, dur):
            t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
            # Gentle fade to avoid clicks; small overtone for color
            wave = (
                0.65 * np.sin(2 * np.pi * freq * t) +
                0.20 * np.sin(2 * np.pi * (2 * freq) * t)
            )
            # Apply short attack/release envelope
            attack = int(0.01 * SAMPLE_RATE)
            release = int(0.03 * SAMPLE_RATE)
            env = np.ones_like(wave)
            if attack > 0:
                env[:attack] = np.linspace(0.0, 1.0, attack)
            if release > 0:
                env[-release:] = np.linspace(1.0, 0.0, release)
            wave = 0.6 * wave * env
            snd = make_sound_from_wave(wave, volume=0.8)
            return snd

        def _key_at_pos(self, pos):
            # Black keys are on top visually
            for k in self.black_keys:
                if k["rect"].collidepoint(pos):
                    return k
            for k in self.white_keys:
                if k["rect"].collidepoint(pos):
                    return k
            return None

        def handle_event(self, event):
            if event.type == pygame.MOUSEMOTION:
                pos = event.pos
                self.hover_power = self.power_rect.collidepoint(pos)
                k = self._key_at_pos(pos)
                self.hover_key = k["name"] if k else None

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if self.power_rect.collidepoint(pos):
                    self.power_on = not self.power_on
                else:
                    k = self._key_at_pos(pos)
                    if k and self.power_on:
                        self._play(k["name"])
                        self.active_times[k["name"]] = 0.12

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.power_on = False
                    for n in self.active_times:
                        self.active_times[n] = 0.0

        def _play(self, name):
            snd = self.sounds.get(name)
            if snd:
                snd.play()

        def update(self, dt):
            # decay active highlights
            for n in self.active_times:
                if self.active_times[n] > 0:
                    self.active_times[n] = max(0.0, self.active_times[n] - dt)

            # If hover not updated by events (e.g., window focus), check mouse
            pos = pygame.mouse.get_pos()
            self.hover_power = self.power_rect.collidepoint(pos)
            k = self._key_at_pos(pos)
            self.hover_key = k["name"] if k else None

        def draw(self, screen_):
            # Draw decorative sprite if provided
            if self.sprite and "surface" in self.sprite and "rect" in self.sprite:
                screen_.blit(self.sprite["surface"], self.sprite["rect"])

            # Keys frame outline
            pygame.draw.rect(screen_, (0, 0, 0), self.keys_frame, width=2, border_radius=8)

            # Draw hover/active overlays for white keys
            for k in self.white_keys:
                name = k["name"]
                rect = k["rect"]
                # Visual separators for white keys
                pygame.draw.rect(screen_, (0, 0, 0), rect, width=1)
                # Active or hover tint
                if self.active_times[name] > 0 or self.hover_key == name:
                    alpha = 90 if self.hover_key == name else 70
                    if self.active_times[name] > 0:
                        alpha = 120
                    tint = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                    tint.fill((40, 140, 255, alpha))
                    screen_.blit(tint, rect.topleft)

            # Draw hover/active overlays for black keys (on top)
            for k in self.black_keys:
                name = k["name"]
                rect = k["rect"]
                # Outline for clarity
                pygame.draw.rect(screen_, (0, 0, 0), rect, width=1)
                if self.active_times[name] > 0 or self.hover_key == name:
                    alpha = 110 if self.hover_key == name else 90
                    if self.active_times[name] > 0:
                        alpha = 150
                    tint = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                    tint.fill((40, 140, 255, alpha))
                    screen_.blit(tint, rect.topleft)

            # Power button
            pr = self.power_rect
            col_on = (40, 170, 80)
            col_off = (170, 60, 60)
            col = col_on if self.power_on else col_off
            base = (230, 230, 230)
            pygame.draw.rect(screen_, base, pr, border_radius=12)
            pygame.draw.rect(screen_, col, pr, width=3, border_radius=12)
            if self.hover_power:
                glow = pygame.Surface((pr.width, pr.height), pygame.SRCALPHA)
                glow.fill((*col, 50))
                screen_.blit(glow, pr.topleft)
            label = "ON" if self.power_on else "OFF"
            txt = self.font_med.render(label, True, col)
            screen_.blit(txt, txt.get_rect(center=pr.center))

            # HUD
            hud = self.font_small.render("Click ON, then click keys • R to reset • ESC to quit", True, (20, 20, 20))
            screen_.blit(hud, (10, 10))

        # End PianoGame

    return PianoGame()
