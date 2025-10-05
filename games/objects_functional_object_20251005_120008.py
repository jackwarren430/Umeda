import pygame
import numpy as np

def create_game(screen, background_surf, context):
    """
    Interactive piano: keys only sound when the power button is on.
    """
    SAMPLE_RATE = context["audio"]["SAMPLE_RATE"]
    make_sound_from_wave = context["audio"]["make_sound_from_wave"]

    W, H = screen.get_size()
    graph = context.get("graph", {})

    def rect_from_graph(node_id, default):
        try:
            for n in graph.get("nodes", []):
                if n.get("id") == node_id and "bbox" in n:
                    b = n["bbox"]
                    return pygame.Rect(b["x"], b["y"], b["w"], b["h"])
        except Exception:
            pass
        return pygame.Rect(*default)

    # Fallback layout if graph missing
    kx, ky, kw, kh = (W // 2 - 280, H // 2 - 80, 560, 190)
    white_w = kw // 7
    white_h = kh
    kb_frame_rect = rect_from_graph("keyboard-frame", (kx, ky, kw, kh))
    # Default white keys
    white_defaults = [(kb_frame_rect.x + i * white_w, kb_frame_rect.y, white_w, white_h) for i in range(7)]
    # Default black keys (skip between E-F and B-C on next octave)
    black_offsets = [0.65, 1.65, 3.65, 4.65, 5.65]  # relative to white key width
    black_w = max(18, int(white_w * 0.55))
    black_h = int(white_h * 0.63)
    black_defaults = []
    for off in black_offsets:
        cx = int(kb_frame_rect.x + off * white_w - black_w // 2)
        black_defaults.append((cx, kb_frame_rect.y, black_w, black_h))

    # Regions from graph or fallbacks
    regions_spec = [
        ("white-key-c", "C", False, white_defaults[0]),
        ("white-key-d", "D", False, white_defaults[1]),
        ("white-key-e", "E", False, white_defaults[2]),
        ("white-key-f", "F", False, white_defaults[3]),
        ("white-key-g", "G", False, white_defaults[4]),
        ("white-key-a", "A", False, white_defaults[5]),
        ("white-key-b", "B", False, white_defaults[6]),
        ("black-key-cs", "C#", True, black_defaults[0]),
        ("black-key-ds", "D#", True, black_defaults[1]),
        ("black-key-fs", "F#", True, black_defaults[2]),
        ("black-key-gs", "G#", True, black_defaults[3]),
        ("black-key-as", "A#", True, black_defaults[4]),
    ]

    power_rect_default = (W - 220, 90, 140, 70)
    power_rect = rect_from_graph("power-toggle", power_rect_default)
    title_rect = rect_from_graph("title-label", (kb_frame_rect.x, kb_frame_rect.y - 140, 260, 90))
    hud_rect = rect_from_graph("hint-hud", (W // 2 - 180, H - 50, 360, 26))

    # Build region objects
    class KeyRegion:
        __slots__ = ("id", "note", "is_black", "rect")
        def __init__(self, _id, note, is_black, rect):
            self.id = _id
            self.note = note
            self.is_black = is_black
            self.rect = rect

    keys = []
    for _id, note, is_black, dflt in regions_spec:
        keys.append(KeyRegion(_id, note, is_black, rect_from_graph(_id, dflt)))

    # Hover priority: black keys above whites
    black_keys = [k for k in keys if k.is_black]
    white_keys = [k for k in keys if not k.is_black]
    all_keys_draw_order = white_keys + black_keys  # draw whites under blacks
    hover_order = black_keys + white_keys          # hit-test blacks first

    # Note frequencies (A4=440 Hz)
    midi_map = {"C":60, "C#":61, "D":62, "D#":63, "E":64, "F":65, "F#":66, "G":67, "G#":68, "A":69, "A#":70, "B":71}
    def note_freq(name):
        n = midi_map[name]
        return 440.0 * (2.0 ** ((n - 69) / 12.0))

    def play_tone(freq, dur=0.35, vol=0.8):
        t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
        wave = 0.6 * np.sin(2 * np.pi * freq * t) * np.linspace(1, 0.9, t.size)
        snd = make_sound_from_wave(wave, volume=vol)
        snd.play()

    # State
    power_on = False
    hover_id = None
    mouse_down = False
    last_drag_key = None
    flash = {}  # id -> remaining_time
    sprite = context.get("sprite")

    # Fonts
    try:
        font_title = pygame.font.SysFont("Arial", 64, bold=True)
        font_ui = pygame.font.SysFont("Arial", 28, bold=True)
        font_hud = pygame.font.SysFont("Arial", 18)
    except Exception:
        font_title = pygame.font.Font(None, 64)
        font_ui = pygame.font.Font(None, 28)
        font_hud = pygame.font.Font(None, 18)

    def region_at(pos):
        # Power button
        if power_rect.collidepoint(pos):
            return "power-toggle"
        # Keys (black first)
        for k in hover_order:
            if k.rect.collidepoint(pos):
                return k.id
        return None

    def trigger_key(key_obj):
        if not power_on:
            return
        freq = note_freq(key_obj.note)
        play_tone(freq)
        flash[key_obj.id] = 0.15

    class Game:
        def handle_event(self, event):
            nonlocal power_on, hover_id, mouse_down, last_drag_key
            if event.type == pygame.MOUSEMOTION:
                hover_id = region_at(event.pos)
                if mouse_down and power_on:
                    # Drag over new keys triggers sound
                    for k in hover_order:
                        if k.rect.collidepoint(event.pos):
                            if last_drag_key != k.id:
                                trigger_key(k)
                                last_drag_key = k.id
                            break
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_down = True
                hover_id = region_at(event.pos)
                if hover_id == "power-toggle":
                    power_on = not power_on
                else:
                    for k in hover_order:
                        if k.rect.collidepoint(event.pos):
                            trigger_key(k)
                            last_drag_key = k.id
                            break
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mouse_down = False
                last_drag_key = None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                elif event.key == pygame.K_r:
                    flash.clear()
                    last_drag_key = None
                    mouse_down = False
                    # Reset power (off by default)
                    power_on = False
                    # Stop any currently playing audio
                    try:
                        pygame.mixer.stop()
                    except Exception:
                        pass

        def update(self, dt):
            # Update flash timers
            to_del = []
            for k, t in flash.items():
                t -= dt
                if t <= 0:
                    to_del.append(k)
                else:
                    flash[k] = t
            for k in to_del:
                del flash[k]

        def draw(self, scr):
            # Background already drawn by shell.

            # Optional sprite overlay if provided
            if sprite and isinstance(sprite, dict) and "surface" in sprite and "rect" in sprite:
                scr.blit(sprite["surface"], sprite["rect"])

            # Draw keyboard frame
            pygame.draw.rect(scr, (0, 0, 0), kb_frame_rect, width=4, border_radius=14)

            # Draw white keys highlights and outlines
            for k in white_keys:
                r = k.rect
                # subtle shadow base
                base = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
                base.fill((255, 255, 255, 20))
                scr.blit(base, r.topleft)

                # highlight on hover/flash
                alpha = 0
                if hover_id == k.id:
                    alpha = 80
                if k.id in flash:
                    alpha = max(alpha, 120)
                if alpha > 0:
                    hsurf = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
                    hsurf.fill((200, 230, 255, alpha))
                    scr.blit(hsurf, r.topleft)

                pygame.draw.rect(scr, (0, 0, 0), r, width=2, border_radius=6)

            # Draw black keys as darker overlays (only highlights so as not to hide art)
            for k in black_keys:
                r = k.rect
                # base dark overlay
                base = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
                base.fill((0, 0, 0, 140))
                scr.blit(base, r.topleft)
                # hover/flash lighter
                alpha = 0
                if hover_id == k.id:
                    alpha = 80
                if k.id in flash:
                    alpha = max(alpha, 120)
                if alpha > 0:
                    hsurf = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
                    hsurf.fill((80, 160, 255, alpha))
                    scr.blit(hsurf, r.topleft)
                pygame.draw.rect(scr, (0, 0, 0), r, width=2, border_radius=4)

            # Power button
            col_bg = (30, 190, 90) if power_on else (200, 200, 200)
            col_border = (0, 120, 60) if power_on else (60, 60, 60)
            pr = power_rect.copy()
            pygame.draw.rect(scr, (255, 255, 255), pr, border_radius=16)
            pygame.draw.rect(scr, col_bg, pr.inflate(-10, -10), border_radius=12)
            bw = 4 if (hover_id == "power-toggle") else 3
            pygame.draw.rect(scr, col_border, pr, width=bw, border_radius=16)
            # Power text
            label = "on" if power_on else "off"
            txt = font_ui.render(label, True, (0, 0, 0))
            scr.blit(txt, txt.get_rect(center=pr.center))

            # Title and HUD
            title_surf = font_title.render("Piano", True, (0, 0, 0))
            scr.blit(title_surf, title_surf.get_rect(midleft=(title_rect.x + 10, title_rect.centery)))

            hud_text = "Click keys • ESC to quit • R to reset"
            hud_surf = font_hud.render(hud_text, True, (20, 20, 20))
            scr.blit(hud_surf, hud_surf.get_rect(center=(hud_rect.centerx, hud_rect.centery)))

    return Game()
