import sys
import os
import math
import pygame
import numpy as np

SAMPLE_RATE = 44100

def init_audio(force_channels=1):
    import pygame
    try:
        pygame.mixer.quit()
    except Exception:
        pass
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=force_channels, buffer=512)
    # optional: print("Mixer:", pygame.mixer.get_init())

def make_sound_from_wave(wave_float, volume=0.6):
    
    # wave_float: 1-D numpy array in [-1.0, 1.0]
    # Returns a pygame.Sound shaped to the current mixer channels.
    
    import numpy as np, pygame
    # int16 PCM
    mono = (np.clip(wave_float * volume, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16)
    mi = pygame.mixer.get_init()  # (freq, size, channels) or None
    channels = mi[2] if mi else 2
    if channels == 1:
        audio = mono
    else:
        audio = np.repeat(mono[:, None], channels, axis=1)  # (n, channels)
    return pygame.sndarray.make_sound(audio)

def midi_to_freq(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))

def synth_wave_for_midi(midi, dur=0.7):
    freq = midi_to_freq(midi)
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    # Simple additive: sine + weak second harmonic
    wave = 0.85*np.sin(2*np.pi*freq*t) + 0.15*np.sin(2*np.pi*2*freq*t)
    # ADSR envelope (short attack/release)
    attack = max(1, int(0.01 * SAMPLE_RATE))
    release = max(1, int(0.08 * SAMPLE_RATE))
    sustain_len = max(1, len(t) - attack - release)
    env = np.concatenate([
        np.linspace(0, 1, attack, endpoint=False),
        np.ones(sustain_len),
        np.linspace(1, 0, release, endpoint=False)
    ])
    env = env[:len(t)]
    wave *= env
    # Normalize a bit
    wave /= np.max(np.abs(wave) + 1e-6)
    return wave.astype(np.float32)

class Key:
    def __init__(self, rect, midi, name, is_black):
        self.rect = rect
        self.midi = midi
        self.name = name
        self.is_black = is_black

def detect_keyboard_rect(surface):
    # Try to detect the inner keyboard region automatically; fall back to a heuristic box
    arr3 = pygame.surfarray.array3d(surface)  # (w, h, 3)
    arr3 = np.transpose(arr3, (1, 0, 2))      # (h, w, 3)
    gray = (0.2126*arr3[:,:,0] + 0.7152*arr3[:,:,1] + 0.0722*arr3[:,:,2]).astype(np.float32)

    h, w = gray.shape
    # Threshold for dark pixels (adaptive)
    thr = float(np.clip(np.percentile(gray, 25), 30, 140))
    dark = (gray < thr).astype(np.float32)

    # Horizontal density
    row_sum = dark.sum(axis=1)
    k = max(5, h // 50)
    kernel = np.ones(k, dtype=np.float32) / k
    row_sm = np.convolve(row_sum, kernel, mode='same')
    r_peak = int(np.argmax(row_sm))
    r_threshold = row_sm.max() * 0.5
    top = r_peak
    while top > 0 and row_sm[top] > r_threshold:
        top -= 1
    bottom = r_peak
    while bottom < h-1 and row_sm[bottom] > r_threshold:
        bottom += 1
    top = max(0, top - int(0.02*h))
    bottom = min(h-1, bottom + int(0.02*h))

    # Vertical density within band
    col_sum = dark[top:bottom+1, :].sum(axis=0)
    k2 = max(5, w // 50)
    kernel2 = np.ones(k2, dtype=np.float32) / k2
    col_sm = np.convolve(col_sum, kernel2, mode='same')
    c_threshold = col_sm.max() * 0.2
    cols = np.where(col_sm > c_threshold)[0]
    if len(cols) > 0:
        left = int(cols[0])
        right = int(cols[-1])
    else:
        left, right = int(0.1*w), int(0.9*w)

    # Shrink margins to avoid outlines
    mx = int(0.03 * w)
    my = int(0.02 * h)
    left = max(0, left + mx)
    right = min(w-1, right - mx)
    top = max(0, top + my)
    bottom = min(h-1, bottom - my)
    rect = pygame.Rect(left, top, max(1, right - left), max(1, bottom - top))

    # Validate; if not plausible, fall back to centered box at lower half
    if rect.w < rect.h * 1.2 or rect.h < h * 0.1 or rect.w < w * 0.2:
        kb_w = int(w * 0.75)
        kb_h = int(h * 0.28)
        kb_x = (w - kb_w) // 2
        kb_y = int(h * 0.58)
        rect = pygame.Rect(kb_x, kb_y, kb_w, kb_h)
    return rect

def build_piano_keys(krect):
    # 7 white keys: C D E F G A B; 5 black keys: C# D# F# G# A#
    names_white = ['C4','D4','E4','F4','G4','A4','B4']
    midis_white = [60,62,64,65,67,69,71]
    names_black = ['C#4','D#4','F#4','G#4','A#4']
    midis_black = [61,63,66,68,70]
    keys = []

    white_w = krect.w / 7.0
    # Build white keys
    for i, (n, m) in enumerate(zip(names_white, midis_white)):
        x0 = int(round(krect.left + i * white_w))
        x1 = int(round(krect.left + (i+1) * white_w))
        rect = pygame.Rect(x0, krect.top, max(1, x1 - x0), krect.h)
        keys.append(Key(rect, m, n, is_black=False))

    # Build black keys (positions between specific white keys)
    black_positions = [0, 1, 3, 4, 5]  # indices before which black keys sit (between i and i+1)
    black_w = int(white_w * 0.6)
    black_h = int(krect.h * 0.6)
    for idx, (n, m) in enumerate(zip(black_positions, midis_black)):
        center_x = krect.left + (idx if idx < 2 else idx+1)  # this mapping is wrong; fix next
    # Proper mapping: use the positions list directly
    black_rects = []
    for pos_idx, (n, m) in zip(black_positions, zip(names_black, midis_black)):
        n_name, m_midi = n  # unpack zipped pair
        # Center at boundary between white key pos_idx and pos_idx+1
        cx = krect.left + (pos_idx + 1) * white_w
        x = int(round(cx - black_w / 2))
        rect = pygame.Rect(x, krect.top, black_w, black_h)
        black_rects.append(Key(rect, m_midi, n_name, is_black=True))

    # Return with black keys last for drawing overlay logic control; but we will check blacks first for hit tests
    return keys, black_rects

def build_piano_keys_fixed(krect):
    # Simpler, correct construction without the mistake above
    names_white = ['C4','D4','E4','F4','G4','A4','B4']
    midis_white = [60,62,64,65,67,69,71]
    names_black = ['C#4','D#4','F#4','G#4','A#4']
    midis_black = [61,63,66,68,70]

    keys_white = []
    white_w = krect.w / 7.0
    for i, (n, m) in enumerate(zip(names_white, midis_white)):
        x0 = int(round(krect.left + i * white_w))
        x1 = int(round(krect.left + (i+1) * white_w))
        rect = pygame.Rect(x0, krect.top, max(1, x1 - x0), krect.h)
        keys_white.append(Key(rect, m, n, is_black=False))

    # Black key positions between white keys: 0,1,(skip 2),3,4,5
    black_positions = [0, 1, 3, 4, 5]
    keys_black = []
    black_w = int(white_w * 0.6)
    black_h = int(krect.h * 0.6)
    for pos, n, m in zip(black_positions, names_black, midis_black):
        cx = krect.left + (pos + 1) * white_w
        x = int(round(cx - black_w / 2))
        rect = pygame.Rect(x, krect.top, black_w, black_h)
        keys_black.append(Key(rect, m, n, is_black=True))
    return keys_white, keys_black

def prepare_sounds(midis):
    sounds = {}
    for m in midis:
        wave = synth_wave_for_midi(m, dur=0.8)
        sounds[m] = make_sound_from_wave(wave, volume=0.7)
    return sounds

def draw_overlays(screen, keys_white, keys_black, hover_key):
    # semi-transparent overlays
    # Draw white key outlines lightly, black keys filled darker area to emphasize zones
    for k in keys_white:
        outline_color = (0, 0, 0, 60)
        surf = pygame.Surface((k.rect.w, k.rect.h), pygame.SRCALPHA)
        pygame.draw.rect(surf, outline_color, pygame.Rect(0,0,k.rect.w, k.rect.h), width=2, border_radius=2)
        screen.blit(surf, (k.rect.x, k.rect.y))

    for k in keys_black:
        fill_color = (0, 0, 0, 90)
        surf = pygame.Surface((k.rect.w, k.rect.h), pygame.SRCALPHA)
        pygame.draw.rect(surf, fill_color, pygame.Rect(0,0,k.rect.w, k.rect.h), width=0, border_radius=3)
        screen.blit(surf, (k.rect.x, k.rect.y))

    if hover_key is not None:
        c = (80, 160, 255, 90) if not hover_key.is_black else (255, 220, 80, 110)
        surf = pygame.Surface((hover_key.rect.w, hover_key.rect.h), pygame.SRCALPHA)
        pygame.draw.rect(surf, c, pygame.Rect(0,0,hover_key.rect.w, hover_key.rect.h), border_radius=2)
        screen.blit(surf, (hover_key.rect.x, hover_key.rect.y))

def key_at_pos(keys_white, keys_black, pos):
    # Black keys take precedence if overlapping
    for k in keys_black:
        if k.rect.collidepoint(pos):
            return k
    for k in keys_white:
        if k.rect.collidepoint(pos):
            return k
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py path/to/image.png")
        return

    img_path = sys.argv[1]
    pygame.init()
    init_audio()


    background = pygame.image.load(img_path)            # 1) load WITHOUT convert
    W, H = background.get_size()                        # 2) get size first
    screen = pygame.display.set_mode((W, H))            # 3) set video mode
    # 4) NOW convert to match the display’s pixel format
    background = background.convert_alpha() if background.get_alpha() else background.convert()

    pygame.display.set_caption("Interactive Piano from Image")

    # Detect keyboard rect and create key regions
    kb_rect = detect_keyboard_rect(background)
    keys_white, keys_black = build_piano_keys_fixed(kb_rect)

    # Prepare sounds
    all_midis = [k.midi for k in keys_white + keys_black]
    sound_map = prepare_sounds(all_midis)

    clock = pygame.time.Clock()
    running = True
    hover_key = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    # Optional screenshot
                    pygame.image.save(screen, "screenshot.png")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                k = key_at_pos(keys_white, keys_black, event.pos)
                if k is not None:
                    snd = sound_map.get(k.midi)
                    if snd:
                        snd.play()
                    print(k.name)

        hover_key = key_at_pos(keys_white, keys_black, pygame.mouse.get_pos())

        # Draw
        screen.blit(background, (0, 0))
        draw_overlays(screen, keys_white, keys_black, hover_key)

        # Optional label near detected area
        # pygame.draw.rect(screen, (0,255,0), kb_rect, 1)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
