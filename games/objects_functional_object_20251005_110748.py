import pygame
import math
import numpy as np

def create_game(screen, background_surf, context):
    '''
    Returns an object with methods:
      handle_event(event)
      update(dt)
      draw(screen)
    '''
    audio = context.get("audio", {})
    SAMPLE_RATE = audio.get("SAMPLE_RATE", 44100)
    make_sound_from_wave = audio.get("make_sound_from_wave")

    # Sprite placement (used only as a decorative overlay + to infer interactive regions)
    spr = context.get("sprite")
    if spr:
        spr_surf = spr.get("surface")
        if "rect" in spr and spr["rect"]:
            spr_rect = spr["rect"].copy()
        else:
            m = spr.get("meta", {})
            spr_rect = pygame.Rect(m.get("x", 0), m.get("y", 0), m.get("w", spr_surf.get_width() if spr_surf else 200), m.get("h", spr_surf.get_height() if spr_surf else 200))
    else:
        spr_surf = None
        # Fallback placement if no sprite is provided: center a placeholder rect
        W, H = screen.get_size()
        w, h = int(W * 0.6), int(H * 0.6)
        spr_rect = pygame.Rect((W - w) // 2, (H - h) // 2, w, h)

    # Approximate geometry of the drawing: a switch on the left, a bulb on the right.
    def compute_geometry(rect):
        w, h = rect.size
        # Switch plate (left-side rectangle)
        plate_w, plate_h = int(w * 0.26), int(h * 0.55)
        plate_x = rect.left + int(w * 0.03)
        plate_y = rect.top + int(h * 0.20)
        plate = pygame.Rect(plate_x, plate_y, plate_w, plate_h)

        # Button circle within the plate
        btn_r = int(min(plate_w, plate_h) * 0.22)
        btn_cx = plate.centerx - int(plate_w * 0.10)
        btn_cy = plate.centery
        button = (btn_cx, btn_cy, btn_r)

        # Bulb glass (right-side circle-ish area)
        bulb_cx = rect.left + int(w * 0.74)
        bulb_cy = rect.top + int(h * 0.33)
        bulb_r = int(min(w, h) * 0.16)
        bulb = (bulb_cx, bulb_cy, bulb_r)

        # Bulb base rectangle under the bulb
        base_w = int(bulb_r * 1.1)
        base_h = int(bulb_r * 0.5)
        base = pygame.Rect(bulb_cx - base_w // 2, bulb_cy + int(bulb_r * 0.35), base_w, base_h)

        return plate, button, bulb, base

    plate_rect, (btn_cx, btn_cy, btn_r), (bulb_cx, bulb_cy, bulb_r), bulb_base = compute_geometry(spr_rect)

    # Prebuild a soft glow surface for the bulb
    def make_glow(r):
        size = r * 4
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        center = (size // 2, size // 2)
        # Concentric circles to fake a radial gradient
        for i in range(r, 0, -1):
            alpha = int(200 * (i / r) ** 2)
            color = (255, 230, 80, alpha)
            pygame.draw.circle(surf, color, center, i * 2)
        return surf

    glow_surf = make_glow(max(8, bulb_r))

    # State
    is_on = False
    brightness = 0.0
    target_brightness = 0.0
    hover_button = False
    mouse_pos = (0, 0)

    # Lightweight click sound
    def play_toggle_sound(up=True):
        if not make_sound_from_wave:
            return
        dur = 0.08 if up else 0.06
        freq = 1100 if up else 700
        t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
        env = np.exp(-t * 40)  # fast decay
        wave = 0.6 * np.sin(2 * np.pi * freq * t) * env
        snd = make_sound_from_wave(wave, volume=0.8)
        try:
            snd.play()
        except Exception:
            pass

    def point_in_button(pos):
        x, y = pos
        return (x - btn_cx) ** 2 + (y - btn_cy) ** 2 <= btn_r ** 2

    class Game:
        def handle_event(self, event):
            nonlocal is_on, target_brightness, hover_button, mouse_pos
            if event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos
                hover_button = point_in_button(mouse_pos)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos
                hover_button = point_in_button(mouse_pos)
                if hover_button:
                    is_on = not is_on
                    target_brightness = 1.0 if is_on else 0.0
                    play_toggle_sound(up=is_on)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    is_on = False
                    target_brightness = 0.0

        def update(self, dt):
            nonlocal brightness
            # Smoothly approach target brightness
            speed = 6.0  # higher = snappier
            if abs(brightness - target_brightness) < 1e-3:
                brightness = target_brightness
            else:
                brightness += (target_brightness - brightness) * min(1.0, dt * speed)

        def draw(self, screen_):
            # Draw decorative sprite (lines) if provided
            if spr_surf:
                screen_.blit(spr_surf, spr_rect.topleft)

            # Hover highlight for the button
            if hover_button or brightness > 0.0:
                alpha = 70 if hover_button else 40
                c = (255, 220, 120, alpha)
                overlay = pygame.Surface((btn_r * 2 + 6, btn_r * 2 + 6), pygame.SRCALPHA)
                pygame.draw.circle(overlay, c, (overlay.get_width() // 2, overlay.get_height() // 2), btn_r + 2)
                screen_.blit(overlay, (btn_cx - overlay.get_width() // 2, btn_cy - overlay.get_height() // 2))

            # Lamp glow when on
            if brightness > 0.01:
                # Soft ambient glow around bulb
                g = glow_surf.copy()
                g.set_alpha(int(180 * brightness))
                screen_.blit(g, (bulb_cx - g.get_width() // 2, bulb_cy - g.get_height() // 2), special_flags=0)

                # Bright core in the glass
                core_r = max(4, int(bulb_r * 0.6))
                core_surf = pygame.Surface((core_r * 2, core_r * 2), pygame.SRCALPHA)
                inner = (255, 245, 180, int(220 * brightness))
                pygame.draw.circle(core_surf, inner, (core_r, core_r), core_r)
                screen_.blit(core_surf, (bulb_cx - core_r, bulb_cy - core_r))

                # Light spill on bulb base
                base_overlay = pygame.Surface(bulb_base.size, pygame.SRCALPHA)
                base_color = (255, 235, 120, int(120 * brightness))
                base_overlay.fill(base_color)
                screen_.blit(base_overlay, bulb_base.topleft)

            # Minimal HUD
            hud = "Click the button to toggle light • R to reset • ESC to quit"
            try:
                font = pygame.font.SysFont(None, max(16, int(screen_.get_height() * 0.03)))
                txt = font.render(hud, True, (20, 20, 20))
                pad = 8
                bg_rect = txt.get_rect(topleft=(pad, pad))
                bg = pygame.Surface((bg_rect.width + 10, bg_rect.height + 6), pygame.SRCALPHA)
                bg.fill((255, 255, 255, 180))
                screen_.blit(bg, (bg_rect.left - 5, bg_rect.top - 3))
                screen_.blit(txt, bg_rect.topleft)
            except Exception:
                pass

    return Game()
