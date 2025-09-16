import pygame
import numpy as np


def create_game(screen, background_surf, context):
    class SolarSystemGame:
        def __init__(self, screen, background_surf, context):
            self.screen = screen
            self.background_surf = background_surf
            self.w, self.h = self.screen.get_size()

            # Layout based on image proportions
            smin = min(self.w, self.h)
            self.center = np.array([self.w * 0.5, self.h * 0.55], dtype=float)
            self.orbit_r = smin * 0.28
            self.sun_r = max(8, int(smin * 0.06))
            self.planet_r = max(6, int(smin * 0.035))

            # Motion
            self.angle = 0.0  # radians, 0 at the rightmost point
            self.speed = np.deg2rad(45.0)  # rad/s
            self.moving = False
            self.dragging = False
            self.prev_angle = self.angle

            # "move" button position near top-right as in the image
            margin = int(smin * 0.04)
            bw = int(self.w * 0.18)
            bh = int(self.h * 0.09)
            self.move_rect = pygame.Rect(self.w - bw - margin, margin, bw, bh)

            # Audio helpers
            self.audio = context.get("audio", {})
            self.SAMPLE_RATE = self.audio.get("SAMPLE_RATE", 44100)
            self.make_sound_from_wave = self.audio.get("make_sound_from_wave", None)

            # UI states
            self.hover_move = False

        # -------------------- Audio --------------------
        def play_tone(self, freq=440.0, dur=0.12, volume=0.8):
            try:
                if self.make_sound_from_wave is None:
                    return
                t = np.linspace(0, dur, int(self.SAMPLE_RATE * dur), endpoint=False)
                wave = 0.6 * np.sin(2 * np.pi * freq * t)

                # Short fade in/out to avoid clicks
                fade_t = min(0.01, dur * 0.25)
                nfade = max(1, int(self.SAMPLE_RATE * fade_t))
                if nfade * 2 < wave.size:
                    fade_in = np.linspace(0.0, 1.0, nfade, endpoint=True)
                    fade_out = np.linspace(1.0, 0.0, nfade, endpoint=True)
                    wave[:nfade] *= fade_in
                    wave[-nfade:] *= fade_out

                snd = self.make_sound_from_wave(wave, volume=volume)
                snd.play()
            except Exception:
                # Fail silently if audio cannot be created/played
                pass

        # -------------------- Helpers --------------------
        def planet_pos(self):
            x = self.center[0] + self.orbit_r * np.cos(self.angle)
            y = self.center[1] + self.orbit_r * np.sin(self.angle)
            return np.array([x, y], dtype=float)

        def set_angle_from_pos(self, pos):
            dx = pos[0] - self.center[0]
            dy = pos[1] - self.center[1]
            self.angle = float(np.arctan2(dy, dx))

        # -------------------- Event Handling --------------------
        def handle_event(self, event):
            if event.type == pygame.MOUSEMOTION:
                self.hover_move = self.move_rect.collidepoint(event.pos)
                if self.dragging:
                    self.set_angle_from_pos(event.pos)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self.move_rect.collidepoint(event.pos):
                        self.moving = not self.moving
                        self.play_tone(880 if self.moving else 440, 0.09, 0.8)
                    else:
                        # Start dragging if clicking the planet
                        if np.linalg.norm(self.planet_pos() - np.array(event.pos)) <= self.planet_r + 4:
                            self.dragging = True
                            self.play_tone(720, 0.06, 0.6)
                elif event.button in (4, 5):
                    # Mouse wheel to adjust speed
                    delta = 1 if event.button == 4 else -1
                    self.speed = float(np.clip(self.speed + delta * np.deg2rad(10), np.deg2rad(5), np.deg2rad(240)))
                    self.play_tone(550 + 40 * delta, 0.05, 0.5)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and self.dragging:
                    self.dragging = False
                    self.play_tone(640, 0.05, 0.6)

        # -------------------- Update --------------------
        def update(self, dt):
            if self.moving and not self.dragging:
                self.angle += self.speed * dt

            # Keep angle in [-pi, pi] for robust crossing checks
            a = self.angle
            if a > np.pi or a < -np.pi:
                a = (a + np.pi) % (2 * np.pi) - np.pi
                self.angle = float(a)

            # Beep each time the planet passes the rightmost point (angle crosses 0 going positive)
            if self.moving:
                if self.prev_angle < 0.0 <= self.angle:
                    self.play_tone(660, 0.08, 0.6)
            self.prev_angle = self.angle

        # -------------------- Draw --------------------
        def draw(self, screen):
            # Background is already blitted by the shell.

            # Orbit path (overlay to match/strengthen the image's path)
            pygame.draw.circle(screen, (0, 0, 0), self.center.astype(int), int(self.orbit_r), width=2)

            # Sun (overlay to ensure visibility)
            pygame.draw.circle(screen, (0, 0, 0), self.center.astype(int), int(self.sun_r))

            # Planet
            pp = self.planet_pos().astype(int)
            pygame.draw.circle(screen, (0, 0, 0), pp, int(self.planet_r))
            if self.moving or self.dragging:
                pygame.draw.circle(screen, (50, 150, 255), pp, int(self.planet_r + 3), width=2)

            # Move button overlay
            self._draw_move_button(screen)

        def _draw_move_button(self, screen):
            rect = self.move_rect
            radius = rect.height // 2
            # Fill and border
            base_col = (230, 230, 230)
            active_col = (200, 240, 200)
            hover_tint = (15, 15, 15) if self.hover_move else (0, 0, 0)

            fill_col = active_col if self.moving else base_col
            pygame.draw.rect(screen, fill_col, rect, border_radius=radius)
            pygame.draw.rect(screen, (0, 0, 0), rect, width=3, border_radius=radius)

            # Text label
            try:
                font = pygame.font.SysFont(None, max(14, int(rect.height * 0.6)), bold=True)
            except Exception:
                font = pygame.font.Font(None, max(14, int(rect.height * 0.6)))
            label = "pause" if self.moving else "move"
            text = font.render(label, True, (0, 0, 0))
            text_rect = text.get_rect(center=rect.center)
            # Slight hover offset for subtle feedback
            if self.hover_move:
                text_rect.move_ip(0, -1)
            screen.blit(text, text_rect)

    return SolarSystemGame(screen, background_surf, context)
