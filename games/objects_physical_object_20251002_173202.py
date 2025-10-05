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
    class BouncyBallGame:
        def __init__(self, screen, context):
            assert "sprite" in context and context["sprite"], "sprite required in context"
            spr = context["sprite"]
            self.surf = spr["surface"]
            self.rect = spr["rect"].copy()
            self.initial_rect = self.rect.copy()

            # Physics state (track center as floats)
            self.posx, self.posy = float(self.rect.centerx), float(self.rect.centery)
            self.vx, self.vy = 0.0, 0.0

            # Parameters
            self.gravity_on = True
            self.gravity = 1800.0  # px/s^2
            self.restitution = 0.82  # bounciness
            self.linear_drag = 1.2   # per second coefficient for air drag
            self.pause = False

            # Dragging/throwing
            self.dragging = False
            self.drag_offset = (0.0, 0.0)
            self.drag_samples = []  # (t_sec, x, y)
            self.sample_window = 0.15  # seconds to consider when computing throw

            # Bounds
            self.screen_rect = screen.get_rect()
            self.halfw = self.rect.width * 0.5
            self.halfh = self.rect.height * 0.5

            # Audio
            self.audio_ok = False
            self.last_bounce_sound_t = 0
            try:
                audio = context.get("audio", {})
                self.SAMPLE_RATE = audio["SAMPLE_RATE"]
                self.make_sound_from_wave = audio["make_sound_from_wave"]
                self.audio_ok = True
            except Exception:
                self.audio_ok = False

        # -------------- Helpers --------------
        def _apply_bounds_and_bounce(self):
            bounced = False
            now_ms = pygame.time.get_ticks()

            # Left/Right
            minx = self.screen_rect.left + self.halfw
            maxx = self.screen_rect.right - self.halfw
            if self.posx < minx:
                self.posx = minx
                if self.vx < 0:
                    self.vx = -self.vx * self.restitution
                    bounced = True
            elif self.posx > maxx:
                self.posx = maxx
                if self.vx > 0:
                    self.vx = -self.vx * self.restitution
                    bounced = True

            # Top/Bottom
            miny = self.screen_rect.top + self.halfh
            maxy = self.screen_rect.bottom - self.halfh
            if self.posy < miny:
                self.posy = miny
                if self.vy < 0:
                    self.vy = -self.vy * self.restitution
                    bounced = True
            elif self.posy > maxy:
                self.posy = maxy
                if self.vy > 0:
                    self.vy = -self.vy * self.restitution
                    bounced = True
                    # Tiny ground friction when touching floor
                    self.vx *= 0.985
                    if abs(self.vy) < 18:
                        self.vy = 0.0
                    if abs(self.vx) < 12:
                        self.vx = 0.0

            if bounced:
                speed = math.hypot(self.vx, self.vy)
                if self.audio_ok and speed > 120 and now_ms - self.last_bounce_sound_t > 40:
                    self._play_bounce_sound(speed)
                    self.last_bounce_sound_t = now_ms

        def _play_bounce_sound(self, speed):
            try:
                # Simple sine "thup"
                freq = 220 + min(900, speed * 0.6)
                dur = 0.035 + min(0.07, speed / 5000.0)
                t = np.linspace(0, dur, int(self.SAMPLE_RATE * dur), endpoint=False)
                # Small exponential fade and little pitch drop
                pitch_drop = np.linspace(1.0, 0.92, t.size)
                wave = 0.6 * np.sin(2 * np.pi * (freq * pitch_drop) * t) * np.linspace(1.0, 0.85, t.size)
                snd = self.make_sound_from_wave(wave, volume=0.8)
                snd.play()
            except Exception:
                pass

        def _record_drag_sample(self, mx, my):
            t = pygame.time.get_ticks() / 1000.0
            self.drag_samples.append((t, float(mx), float(my)))
            # Keep only recent samples
            while self.drag_samples and t - self.drag_samples[0][0] > self.sample_window:
                self.drag_samples.pop(0)

        def _throw_velocity_from_drag(self):
            # Estimate average velocity over recent window
            if len(self.drag_samples) < 2:
                return 0.0, 0.0
            t0, x0, y0 = self.drag_samples[0]
            t1, x1, y1 = self.drag_samples[-1]
            dt = max(1e-4, t1 - t0)
            vx = (x1 - x0) / dt
            vy = (y1 - y0) / dt
            # Scale to feel right
            scale = 1.05
            vx *= scale
            vy *= scale
            # Clamp to sane max speed
            max_speed = 2400.0
            speed = math.hypot(vx, vy)
            if speed > max_speed:
                s = max_speed / speed
                vx *= s
                vy *= s
            return vx, vy

        def _reset(self):
            self.rect = self.initial_rect.copy()
            self.posx, self.posy = float(self.rect.centerx), float(self.rect.centery)
            self.vx, self.vy = 0.0, 0.0
            self.pause = False

        # -------------- Public API --------------
        def handle_event(self, event):
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    self.gravity_on = not self.gravity_on
                elif event.key == pygame.K_r:
                    self._reset()
                elif event.key == pygame.K_p:
                    self.pause = not self.pause
                elif event.key == pygame.K_LEFT:
                    self.vx -= 220.0
                elif event.key == pygame.K_RIGHT:
                    self.vx += 220.0
                elif event.key == pygame.K_UP:
                    self.vy -= 220.0
                elif event.key == pygame.K_DOWN:
                    self.vy += 220.0

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.rect.collidepoint(event.pos):
                    self.dragging = True
                    mx, my = event.pos
                    self.drag_offset = (mx - self.posx, my - self.posy)
                    self.vx = self.vy = 0.0
                    self.drag_samples.clear()
                    self._record_drag_sample(mx, my)

            elif event.type == pygame.MOUSEMOTION and self.dragging:
                mx, my = event.pos
                ox, oy = self.drag_offset
                self.posx = float(mx - ox)
                self.posy = float(my - oy)
                self._apply_bounds_and_bounce()  # clamp while dragging
                self._record_drag_sample(mx, my)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
                self.dragging = False
                vx, vy = self._throw_velocity_from_drag()
                self.vx, self.vy = vx, vy
                self.drag_samples.clear()

        def update(self, dt):
            # dt: seconds
            if self.pause:
                # Keep rect in sync if window size changes
                self.rect.center = (round(self.posx), round(self.posy))
                return

            if not self.dragging:
                if self.gravity_on:
                    self.vy += self.gravity * dt

                # Air drag
                drag_factor = math.exp(-self.linear_drag * dt)
                self.vx *= drag_factor
                self.vy *= drag_factor

                # Integrate position
                self.posx += self.vx * dt
                self.posy += self.vy * dt

                # Collisions with window edges
                self._apply_bounds_and_bounce()

                # Sleep small velocities to stop jitter
                if abs(self.vx) < 5:
                    self.vx = 0.0
                if abs(self.vy) < 5 and abs(self.posy - (self.screen_rect.bottom - self.halfh)) < 0.5:
                    self.vy = 0.0

            # Sync rect with float center
            self.rect.center = (round(self.posx), round(self.posy))

        def draw(self, screen):
            # background already drawn by the environment
            screen.blit(self.surf, self.rect)

    return BouncyBallGame(screen, context)
