"""
game_shell.py — Pygame boilerplate runner

Usage:
    python game_shell.py path/to/background.png [--module path/to/game_objects.py]

Contract for the *game objects* module (the piece GPT will generate later):
    - It must define a function:
          def create_game(screen, background_surf, context) -> GameLike
      where the returned object implements:
          handle_event(event)    # called for each pygame.event.get()
          update(dt)             # dt in seconds
          draw(screen)           # draw overlays/objects; background is already blitted

    - 'context' is a dict you can use (e.g., audio helpers below).

This boilerplate:
    - Loads the background, sets the window size to match.
    - Dynamically imports your module by *file path* (or runs a safe fallback demo).
    - Provides robust audio helpers (mono/stereo safe) via context["audio"].
"""

from __future__ import annotations
import sys, os, importlib.util, time, tempfile
import pygame

# ---------- Audio helpers (safe for mono/stereo mixers) ----------
try:
    import numpy as np
except Exception:
    np = None  # Only required if your game code uses synthesized audio

SAMPLE_RATE = 44100

def init_audio(force_channels=1):
    try:
        pygame.mixer.quit()
    except Exception:
        pass
    try:
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=force_channels, buffer=512)
    except Exception as e:
        print("Audio init failed (continuing without audio):", e)

def make_sound_from_wave(wave_float, volume=0.6):
    """
    wave_float: 1-D numpy array in [-1.0, 1.0]
    Returns a pygame.Sound shaped to the current mixer channels.
    """
    if np is None:
        raise RuntimeError("numpy required for audio synthesis")
    mono = (np.clip(wave_float * volume, -1.0, 1.0) * (2**15 - 1)).astype("int16")
    mi = pygame.mixer.get_init()  # (freq, size, channels)
    channels = mi[2] if mi else 2
    if channels == 1:
        audio = mono
    else:
        audio = np.repeat(mono[:, None], channels, axis=1)
    return pygame.sndarray.make_sound(audio)

# ---------- Module loading ----------
def load_game_module(module_path: str):
    """
    Import a Python file by path and return the module object.
    """
    spec = importlib.util.spec_from_file_location("game_objects_dyn", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# ---------- Fallback demo if no module is supplied ----------
class _FallbackGame:
    def __init__(self, screen, background, ctx):
        self.bg = background
        self.points = []
        self.font = pygame.font.SysFont(None, 20)
        self.msg = "No game module provided — click to drop dots; ESC to quit"

        # Make a tiny click sound if numpy/audio available
        self.click_sound = None
        if np is not None and pygame.mixer.get_init():
            t = np.linspace(0, 0.08, int(SAMPLE_RATE*0.08), endpoint=False)
            wave = 0.25*np.sin(2*np.pi*880*t) * np.linspace(1, 0, t.size)
            try:
                self.click_sound = make_sound_from_wave(wave)
            except Exception:
                self.click_sound = None

    def handle_event(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            self.points.append(e.pos)
            if self.click_sound:
                self.click_sound.play()

    def update(self, dt):  # noqa: ARG002
        pass

    def draw(self, screen):
        # background is already blitted by the shell
        for p in self.points:
            pygame.draw.circle(screen, (255, 70, 70), p, 8)
        txt = self.font.render(self.msg, True, (0, 0, 0))
        screen.blit(txt, (10, 10))

# ---------- Main loop ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: python game_shell.py path/to/background.png [--module path/to/game_objects.py]")
        sys.exit(2)

    img_path = sys.argv[1]
    module_path = None
    if len(sys.argv) >= 4 and sys.argv[2] == "--module":
        module_path = sys.argv[3]

    pygame.init()
    init_audio(force_channels=1)  # robust init; module may also reconfigure if needed

    # Load image BEFORE convert; set mode; THEN convert (display contract)
    try:
        background = pygame.image.load(img_path)
    except Exception as e:
        print("Failed to load image:", e)
        sys.exit(1)

    W, H = background.get_size()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Pygame Boiler")

    background = background.convert_alpha() if background.get_alpha() else background.convert()

    # Prepare context for user code
    context = {
        "audio": {
            "SAMPLE_RATE": SAMPLE_RATE,
            "init_audio": init_audio,
            "make_sound_from_wave": make_sound_from_wave,
        },
        "image_path": img_path,
    }

    # Load game module or fallback
    game = None
    if module_path:
        try:
            mod = load_game_module(module_path)
            if hasattr(mod, "create_game"):
                game = mod.create_game(screen, background, context)  # type: ignore[attr-defined]
            else:
                raise RuntimeError("Module has no create_game(screen, background, context)")
        except Exception as e:
            print(f"Failed to load module '{module_path}': {e}\nUsing fallback demo.")
            game = _FallbackGame(screen, background, context)
    else:
        game = _FallbackGame(screen, background, context)

    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
            else:
                try:
                    game.handle_event(e)
                except Exception as ex:
                    print("handle_event error:", ex)

        try:
            game.update(dt)
        except Exception as ex:
            print("update error:", ex)

        # Draw pass: background first, then game overlays/objects
        screen.blit(background, (0, 0))
        try:
            game.draw(screen)
        except Exception as ex:
            print("draw error:", ex)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
