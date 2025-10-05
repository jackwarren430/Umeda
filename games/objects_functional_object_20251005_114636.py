import math
import pygame
import numpy as np

def create_game(screen, background_surf, context):
    """
    Functional Object: Button -> Lightbulb
    Click the circular button to toggle the bulb on/off. Hover highlights. R resets.
    """

    # Helpers to fetch optional assets from context
    sprite = context.get("sprite")
    sprite_surf = sprite["surface"] if sprite and "surface" in sprite else None
    sprite_rect = sprite["rect"] if sprite and "rect" in sprite else None

    audio_ctx = context.get("audio", {})
    SAMPLE_RATE = audio_ctx.get("SAMPLE_RATE", 44100)
    make_sound_from_wave = audio_ctx.get("make_sound_from_wave")

    graph = context.get("graph", {})

    def get_node_bbox(node_id, default_rect):
        nodes = graph.get("nodes") or []
        for n in nodes:
            if n.get("id") == node_id and "bbox" in n:
                b = n["bbox"]
                return pygame.Rect(b["x"], b["y"], b["w"], b["h"])
        return pygame.Rect(default_rect)

    W, H = screen.get_size()

    # Default layout if graph info is missing
    default_button = (int(W * 0.18), int(H * 0.30), int(W * 0.12), int(H * 0.22))
    default_bulb = (int(W * 0.62), int(H * 0.18), int(W * 0.18), int(H * 0.42))

    button_bbox = get_node_bbox("button-hotspot", default_button)
    bulb_bbox = get_node_bbox("bulb-lamp", default_bulb)
    panel_bbox = get_node_bbox("button-panel",
                               (button_bbox.x - 35, button_bbox.y - 40, button_bbox.w + 70, button_bbox.h + 80))

    # Derive circular hit area from button bbox
    btn_cx = button_bbox.centerx
    btn_cy = button_bbox.centery
    btn_r = int(min(button_bbox.w, button_bbox.h) * 0.45)

    # Bulb glow center and base radius
    bulb_center = (bulb_bbox.centerx, bulb_bbox.y + bulb_bbox.h // 3)
    bulb_glow_r = int(max(bulb_bbox.w, bulb_bbox.h) * 0.55)

    # Precompute radial glow surface (normalized 256x256) and scale when drawing
    def make_radial_surface(radius, color=(255, 220, 100)):
        d = radius * 2
        surf = pygame.Surface((d, d), pygame.SRCALPHA)
        for r in range(radius, 0, -1):
            alpha = int(255 * (r / radius) ** 2)
            col = (*color, int(alpha * 0.36))
            pygame.draw.circle(surf, col, (radius, radius), r)
        return surf

    base_glow_radius = max(64, min(256, bulb_glow_r))
    glow_base = make_radial_surface(base_glow_radius)

    # Simple "click" tone
    def play_click(on=True):
        if not make_sound_from_wave:
            return
        dur = 0.08 if on else 0.06
        freq = 880 if on else 440
        t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
        wave = 0.6 * np.sin(2 * np.pi * freq * t) * np.linspace(1, 0.9, t.size)
        snd = make_sound_from_wave(wave, volume=0.8)
        snd.play()

    # Light hum ping when turned on
    def play_on_ping():
        if not make_sound_from_wave:
            return
        dur = 0.18
        f0, f1 = 660, 520
        t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
        sweep = f0 + (f1 - f0) * (t / dur)
        wave = 0.5 * np.sin(2 * np.pi * sweep * t) * np.linspace(1, 0.85, t.size)
        snd = make_sound_from_wave(wave, volume=0.7)
        snd.play()

    # State
    lit = False
    hover = False
    pressed_flash = 0.0  # seconds
    time_since_on = 0.0

    font = pygame.font.SysFont("Arial", 18)

    def point_in_circle(px, py, cx, cy, r):
        dx, dy = px - cx, py - cy
        return dx * dx + dy * dy <= r * r

    class Game:
        def handle_event(self, event):
            nonlocal hover, lit, pressed_flash, time_since_on
            if event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                hover = point_in_circle(mx, my, btn_cx, btn_cy, btn_r)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if point_in_circle(mx, my, btn_cx, btn_cy, btn_r):
                    lit = not lit
                    pressed_flash = 0.12
                    if lit:
                        time_since_on = 0.0
                        play_on_ping()
                        play_click(True)
                    else:
                        play_click(False)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Reset
                    lit = False
                    pressed_flash = 0.0
                    time_since_on = 0.0

        def update(self, dt):
            nonlocal pressed_flash, time_since_on
            if pressed_flash > 0:
                pressed_flash = max(0.0, pressed_flash - dt)
            if lit:
                time_since_on += dt

        def draw(self, scr):
            # Draw provided sprite as decorative overlay (if any)
            if sprite_surf and sprite_rect:
                scr.blit(sprite_surf, sprite_rect)

            # Draw panel outline (subtle, in case sprite is missing)
            panel_color = (30, 30, 30)
            pygame.draw.rect(scr, panel_color, panel_bbox, width=3, border_radius=8)

            # Button look
            base_col = (230, 230, 230)
            hover_col = (255, 245, 200) if hover else base_col
            press_alpha = int(255 * (pressed_flash / 0.12)) if pressed_flash > 0 else 0

            # Button shadow/backplate
            pygame.draw.circle(scr, (200, 200, 200), (btn_cx + 2, btn_cy + 2), btn_r + 6)
            pygame.draw.circle(scr, (35, 35, 35), (btn_cx, btn_cy), btn_r + 6, 3)
            pygame.draw.circle(scr, hover_col, (btn_cx, btn_cy), btn_r)

            # Button highlight ring on hover or lit
            if hover or lit:
                color = (250, 210, 120) if lit else (150, 200, 255)
                pygame.draw.circle(scr, color, (btn_cx, btn_cy), btn_r + 2, 3)

            # Button press flash
            if press_alpha > 0:
                s = pygame.Surface((btn_r * 2 + 10, btn_r * 2 + 10), pygame.SRCALPHA)
                pygame.draw.circle(s, (255, 255, 255, press_alpha), (s.get_width() // 2, s.get_height() // 2), btn_r)
                scr.blit(s, (btn_cx - s.get_width() // 2, btn_cy - s.get_height() // 2))

            # Bulb drawing guide (if sprite absent, we render a simple bulb)
            if not sprite_surf:
                # Bulb glass
                glass_rect = pygame.Rect(bulb_bbox.x + bulb_bbox.w * 0.2, bulb_bbox.y,
                                         bulb_bbox.w * 0.6, bulb_bbox.h * 0.65)
                pygame.draw.ellipse(scr, (40, 40, 40), glass_rect, 3)
                # Base
                base_rect = pygame.Rect(bulb_bbox.centerx - bulb_bbox.w * 0.18,
                                        bulb_bbox.y + int(bulb_bbox.h * 0.62),
                                        int(bulb_bbox.w * 0.36), int(bulb_bbox.h * 0.16))
                pygame.draw.rect(scr, (40, 40, 40), base_rect, 3, border_radius=6)
                # Filaments
                fc = (bulb_bbox.centerx, bulb_bbox.y + int(bulb_bbox.h * 0.2))
                pygame.draw.line(scr, (40, 40, 40),
                                 (bulb_bbox.centerx - 8, bulb_bbox.y + int(bulb_bbox.h * 0.55)),
                                 (bulb_bbox.centerx - 8, bulb_bbox.y + int(bulb_bbox.h * 0.30)), 3)
                pygame.draw.line(scr, (40, 40, 40),
                                 (bulb_bbox.centerx + 8, bulb_bbox.y + int(bulb_bbox.h * 0.55)),
                                 (bulb_bbox.centerx + 8, bulb_bbox.y + int(bulb_bbox.h * 0.30)), 3)
                pygame.draw.circle(scr, (40, 40, 40), (fc[0], fc[1]), 5, 2)

            # Bulb glow if lit
            if lit:
                # Pulsing intensity
                pulse = 0.65 + 0.35 * math.sin(time_since_on * 4.0)
                scale = max(0.85, min(1.25, 1.0 + 0.08 * math.sin(time_since_on * 3.0)))
                radius = int(base_glow_radius * scale)
                glow = pygame.transform.smoothscale(glow_base, (radius * 2, radius * 2))
                # Tint by multiplying alpha
                glow_mod = glow.copy()
                arr = pygame.surfarray.pixels_alpha(glow_mod)
                arr[:] = (arr * pulse).astype(arr.dtype)
                del arr
                scr.blit(glow_mod, (bulb_center[0] - radius, bulb_center[1] - radius), special_flags=pygame.BLEND_PREMULTIPLIED)

                # Bright core
                core_r = max(8, int(bulb_bbox.w * 0.08))
                pygame.draw.circle(scr, (255, 240, 150, 220), bulb_center, core_r)

            # HUD
            hud_text = "Click button • R to reset"
            text_surf = font.render(hud_text, True, (20, 20, 20))
            # Add a subtle background for readability
            bg = pygame.Surface((text_surf.get_width() + 10, text_surf.get_height() + 6), pygame.SRCALPHA)
            bg.fill((255, 255, 255, 140))
            scr.blit(bg, (16, 12))
            scr.blit(text_surf, (21, 15))

    return Game()
