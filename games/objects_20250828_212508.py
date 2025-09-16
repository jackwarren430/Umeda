import pygame as pygame
import numpy as np

# Module API
def create_game(screen, background_surf, context):
    """
    Returns an object implementing:
      handle_event(event), update(dt), draw(screen)
    """

    class PianoGame:
        def __init__(self, screen, background_surf, context):
            self.screen = screen
            self.bg = background_surf  # already blitted by the shell
            self.audio = context.get("audio", {})
            self.sr = int(self.audio.get("SAMPLE_RATE", 44100))
            self.make_sound_from_wave = self.audio.get("make_sound_from_wave")
            self.W, self.H = self.screen.get_size()

            # Heuristic placement of the keyboard region based on the image layout.
            # The piano keyboard in the provided image sits in the lower half,
            # centered with generous margins. We choose a robust relative rectangle.
            kb_w = int(self.W * 0.76)
            kb_h = int(self.H * 0.30)
            kb_x = int((self.W - kb_w) * 0.5)
            kb_y = int(self.H * 0.55 - kb_h * 0.15)  # a bit above the exact half to match the art
            self.kb_rect = pygame.Rect(kb_x, kb_y, kb_w, kb_h)

            # Build clickable key geometry (one octave C4..C5 white keys + accidentals)
            self.white_notes = [
                ("C4", 261.63),
                ("D4", 293.66),
                ("E4", 329.63),
                ("F4", 349.23),
                ("G4", 392.00),
                ("A4", 440.00),
                ("B4", 493.88),
                ("C5", 523.25),
            ]
            self.black_notes = [
                ("C#4", 277.18),
                ("D#4", 311.13),
                ("F#4", 369.99),
                ("G#4", 415.30),
                ("A#4", 466.16),
            ]
            # Indices of white keys after which a black key appears (C#, D#, F#, G#, A#)
            self.black_after_white_idx = [0, 1, 3, 4, 5]

            self.white_keys, self.black_keys = self._build_key_rects(self.kb_rect)

            # Active highlights: list of dicts {rect, color, time_left}
            self.active = []

            # Pre-make small overlay surfaces
            self.overlay_white = pygame.Surface((1, 1), flags=pygame.SRCALPHA)
            self.overlay_white.fill((255, 215, 0, 90))
            self.overlay_black = pygame.Surface((1, 1), flags=pygame.SRCALPHA)
            self.overlay_black.fill((0, 200, 255, 110))

        def _build_key_rects(self, rect):
            white_w = rect.w // 8
            white_rects = []
            x = rect.x
            for i in range(8):
                # Last key absorbs remainder width
                w = white_w if i < 7 else (rect.x + rect.w - x)
                r = pygame.Rect(x, rect.y, w, rect.h)
                white_rects.append(r)
                x += w

            # Black keys are narrower and shorter, centered over gaps between white keys
            black_rects = []
            black_w = max(8, int(white_w * 0.6))
            black_h = int(rect.h * 0.62)
            for bi, wi in enumerate(self.black_after_white_idx):
                # center above the boundary between white key wi and wi+1
                boundary_x = white_rects[wi].right
                r = pygame.Rect(0, 0, black_w, black_h)
                r.centerx = boundary_x
                r.top = rect.y
                # clamp within keyboard rect horizontally
                if r.left < rect.left:
                    r.left = rect.left
                if r.right > rect.right:
                    r.right = rect.right
                black_rects.append(r)

            return white_rects, black_rects

        def _play_tone(self, freq, dur=0.5):
            if not self.make_sound_from_wave:
                return
            t = np.linspace(0, dur, int(self.sr * dur), endpoint=False)
            wave = 0.6 * np.sin(2 * np.pi * freq * t)
            # Short fade in/out to avoid clicks
            fade_len = max(1, int(0.01 * self.sr))
            if fade_len * 2 < wave.size:
                fade_in = np.linspace(0.0, 1.0, fade_len)
                fade_out = np.linspace(1.0, 0.0, fade_len)
                wave[:fade_len] *= fade_in
                wave[-fade_len:] *= fade_out
            snd = self.make_sound_from_wave(wave, volume=0.8)
            if snd:
                snd.play()

        def _trigger_key(self, rect, is_black, freq):
            # Visual highlight
            color_surf = self.overlay_black if is_black else self.overlay_white
            self.active.append({
                "rect": rect.copy(),
                "surf": pygame.transform.smoothscale(color_surf, (rect.w, rect.h)),
                "time_left": 0.18 if is_black else 0.22,
                "is_black": is_black,
            })
            # Sound
            self._play_tone(freq, 0.35 if is_black else 0.45)

        def handle_event(self, event):
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                # Prioritize black keys (they sit above whites)
                for i, r in enumerate(self.black_keys):
                    if r.collidepoint(pos):
                        name, freq = self.black_notes[i]
                        self._trigger_key(r, True, freq)
                        return
                for i, r in enumerate(self.white_keys):
                    if r.collidepoint(pos):
                        name, freq = self.white_notes[i]
                        self._trigger_key(r, False, freq)
                        return

        def update(self, dt):
            # Fade out active highlights
            for a in self.active:
                a["time_left"] -= dt
            self.active = [a for a in self.active if a["time_left"] > 0]

        def draw(self, screen):
            # Draw subtle overlays to indicate interactive regions without hiding the image
            # White key outlines
            outline_color = (0, 0, 0, 180)
            for r in self.white_keys:
                pygame.draw.rect(screen, (0, 0, 0), r, 1)
            # Black key outlines
            for r in self.black_keys:
                pygame.draw.rect(screen, (0, 0, 0), r, 1)

            # Active highlights
            for a in self.active:
                screen.blit(a["surf"], a["rect"])

    return PianoGame(screen, background_surf, context)
