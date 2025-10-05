import pygame
import numpy as np

def create_game(screen, background_surf, context):
    # Use only pygame and numpy
    SAMPLE_RATE = context["audio"]["SAMPLE_RATE"]
    make_sound_from_wave = context["audio"]["make_sound_from_wave"]

    def synth_tone(freq, dur, volume=0.8):
        # Basic sine tone with short fade to avoid clicks
        t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
        wave = 0.6 * np.sin(2 * np.pi * freq * t)
        fade_len = max(1, int(0.01 * SAMPLE_RATE))  # 10 ms fades
        if fade_len * 2 < wave.size:
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            wave[:fade_len] *= fade_in
            wave[-fade_len:] *= fade_out
        snd = make_sound_from_wave(wave, volume=volume)
        return snd

    class LightSwitchGame:
        def __init__(self, screen, background_surf):
            self.screen = screen
            self.bg = background_surf
            self.w = background_surf.get_width()
            self.h = background_surf.get_height()

            # Detect obvious regions (left switch button and right bulb) from the drawing.
            # Strategy: Threshold dark strokes, then consider left and right halves separately.
            arr = pygame.surfarray.array3d(self.bg)  # (w, h, 3)
            # Convert to (h, w, 3)
            arr = np.transpose(arr, (1, 0, 2)).astype(np.float32)
            lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
            dark = lum < 240  # black stroke on white background

            # Left region (likely the switch/box)
            split_l = int(0.45 * self.w)
            left_mask = dark[:, :split_l]
            right_mask = dark[:, int(0.55 * self.w):]

            def bounding_rect_from_mask(mask, x_offset=0):
                ys, xs = np.where(mask)
                if ys.size == 0:
                    return pygame.Rect(x_offset + int(0.1 * self.w), int(0.2 * self.h), int(0.2 * self.w), int(0.4 * self.h))
                x0 = xs.min() + x_offset
                x1 = xs.max() + x_offset
                y0 = ys.min()
                y1 = ys.max()
                return pygame.Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1)

            self.left_box = bounding_rect_from_mask(left_mask, 0)
            self.right_box = bounding_rect_from_mask(right_mask, int(0.55 * self.w))

            # Define switch button as a circle centered in left_box
            self.button_center = (self.left_box.centerx, self.left_box.centery)
            self.button_radius = int(0.18 * min(self.left_box.width, self.left_box.height))
            if self.button_radius < 10:
                self.button_radius = 18

            # Define bulb "glass" as an ellipse occupying upper 65% of right_box
            rx, ry, rw, rh = self.right_box
            rh_glass = int(0.65 * rh)
            self.bulb_rect = pygame.Rect(rx + int(0.1 * rw), ry, int(0.8 * rw), rh_glass)

            # Wire path approximated by a smooth arc between bottom centers
            self.wire_start = (self.left_box.midbottom[0], self.left_box.bottom + 2)
            self.wire_end = (self.right_box.midbottom[0], self.right_box.bottom + 2)
            midx = (self.wire_start[0] + self.wire_end[0]) / 2
            ctrl_y = int(self.h * 0.85)
            self.wire_ctrl = (midx, ctrl_y)
            self.pulse_phase = 0.0

            # Interaction state
            self.is_on = False
            self.is_pressed_visual = 0.0  # 0..1
            self.target_pressed_visual = 0.0
            self.glow = 0.0

            # Sounds
            self.hum_sound = None

            # Colors
            self.col_off = (90, 90, 90)
            self.col_on = (255, 230, 120)
            self.col_button = (40, 160, 80)
            self.col_button_off = (140, 140, 140)

            # Pre-render shadow/glow helpers
            self.glow_surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)

        # Helpers
        def point_in_circle(self, pos, center, radius):
            dx = pos[0] - center[0]
            dy = pos[1] - center[1]
            return dx*dx + dy*dy <= radius*radius

        def point_in_ellipse(self, pos, rect):
            cx = rect.centerx
            cy = rect.centery
            rx = rect.width / 2.0
            ry = rect.height / 2.0
            if rx <= 0 or ry <= 0:
                return False
            nx = (pos[0] - cx) / rx
            ny = (pos[1] - cy) / ry
            return nx*nx + ny*ny <= 1.0

        def toggle_light(self, desired=None):
            new_state = (not self.is_on) if desired is None else bool(desired)
            if new_state == self.is_on:
                return
            self.is_on = new_state
            # Audio feedback
            if self.is_on:
                synth_tone(800, 0.08, 0.8).play()
                # Start gentle 120 Hz hum
                self.hum_sound = synth_tone(120, 1.5, 0.3)
                try:
                    self.hum_sound.play(loops=-1)
                except TypeError:
                    # Some environments use play(-1)
                    self.hum_sound.play(-1)
            else:
                synth_tone(300, 0.06, 0.7).play()
                if self.hum_sound:
                    self.hum_sound.stop()
                    self.hum_sound = None

        def handle_event(self, event):
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if self.point_in_circle(pos, self.button_center, self.button_radius):
                    self.target_pressed_visual = 1.0
                    self.toggle_light()  # press toggles
                elif self.point_in_ellipse(pos, self.bulb_rect):
                    # Also allow toggling by clicking the bulb
                    self.toggle_light()
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.target_pressed_visual = 0.0

        def update(self, dt):
            # Smooth animation for button press depth
            speed = 8.0
            self.is_pressed_visual += (self.target_pressed_visual - self.is_pressed_visual) * min(1.0, speed * dt)

            # Glow animation
            target_glow = 1.0 if self.is_on else 0.0
            self.glow += (target_glow - self.glow) * min(1.0, 3.0 * dt)

            # Pulse along the wire when on
            if self.is_on:
                self.pulse_phase = (self.pulse_phase + dt * 0.7) % 1.0
            else:
                self.pulse_phase = 0.0

        def draw_glow(self, surface, center, base_radius, intensity):
            # Draw soft radial glow with concentric circles
            steps = 8
            for i in range(steps):
                t = i / (steps - 1)
                r = int(base_radius * (0.35 + 0.78 * t))
                alpha = int(6 + 220 * (1 - t) * (intensity ** 0.9))
                color = (255, 240, 180, max(0, min(255, alpha)))
                pygame.draw.circle(surface, color, center, r)

        def bezier_point(self, t, p0, p1, p2):
            # Quadratic Bezier
            u = 1 - t
            x = u*u*p0[0] + 2*u*t*p1[0] + t*t*p2[0]
            y = u*u*p0[1] + 2*u*t*p1[1] + t*t*p2[1]
            return (int(x), int(y))

        def draw_wire_pulse(self, surface):
            # Draw a small moving dot along the wire to suggest "power flow"
            if not self.is_on:
                return
            # Multiple pulses spaced along the curve
            for k in range(3):
                t = (self.pulse_phase + k * 0.33) % 1.0
                pos = self.bezier_point(t, self.wire_start, self.wire_ctrl, self.wire_end)
                rad = 6
                color = (255, 220, 100)
                pygame.draw.circle(surface, color, pos, rad)
                pygame.draw.circle(surface, (255, 255, 255), pos, max(1, rad // 2))

        def draw_button(self, surface):
            # Visual depth for pressed state
            depth = int(self.button_radius * 0.25 * self.is_pressed_visual)
            cx, cy = self.button_center
            # Outer ring highlight to indicate interactivity
            ring_color = (40, 40, 40)
            pygame.draw.circle(surface, ring_color, (cx, cy), self.button_radius + 6, 3)
            # Button face
            face_color = self.col_button if self.is_on else self.col_button_off
            top = (cx, cy + depth)
            pygame.draw.circle(surface, face_color, top, self.button_radius)
            # Inner shine
            shine_alpha = 90 if self.is_on else 40
            pygame.draw.circle(surface, (255, 255, 255, shine_alpha), (cx - self.button_radius // 4, cy + depth - self.button_radius // 4), max(4, self.button_radius // 2))
            # Outline
            pygame.draw.circle(surface, (0, 0, 0), top, self.button_radius, 3)

        def draw_bulb_glow(self, surface):
            if self.glow <= 0.01:
                return
            # Slight flicker
            time_ms = pygame.time.get_ticks() / 1000.0
            flicker = 0.06 * np.sin(2 * np.pi * 5.3 * time_ms) + 0.03 * np.sin(2 * np.pi * 7.7 * time_ms + 1.2)
            intensity = max(0.0, min(1.0, self.glow + flicker))
            # Center near top of bulb glass
            cx, cy = self.bulb_rect.centerx, int(self.bulb_rect.top + self.bulb_rect.height * 0.45)
            base_radius = int(max(self.bulb_rect.width, self.bulb_rect.height) * 0.55)
            self.glow_surface.fill((0, 0, 0, 0))
            self.draw_glow(self.glow_surface, (cx, cy), base_radius, intensity)
            surface.blit(self.glow_surface, (0, 0))

        def draw(self, screen):
            # Draw interactive overlays only (background is already blitted)
            self.draw_bulb_glow(screen)
            self.draw_wire_pulse(screen)
            self.draw_button(screen)

            # Optional subtle outlines to hint regions
            hover_pos = pygame.mouse.get_pos()
            # Button hover ring
            if self.point_in_circle(hover_pos, self.button_center, self.button_radius + 10):
                pygame.draw.circle(screen, (0, 0, 0), self.button_center, self.button_radius + 10, 1)
            # Bulb hover ellipse
            if self.point_in_ellipse(hover_pos, self.bulb_rect.inflate(20, 20)):
                pygame.draw.ellipse(screen, (0, 0, 0), self.bulb_rect.inflate(20, 20), 1)

    return LightSwitchGame(screen, background_surf)
