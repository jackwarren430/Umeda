import math
import pygame
import numpy as np

def create_game(screen, background_surf, context):
    '''
    Returns an object with methods:
      handle_event(event)    # called for each event
      update(dt)             # dt: float seconds
      draw(screen)           # draw your overlays/sprites; background is already drawn
    '''
    class BouncyBallGame:
        def __init__(self, screen, background_surf, context):
            self.screen = screen
            self.background_surf = background_surf

            # Required sprite elements
            sp = context['sprite']
            self.surf = sp['surface']
            self.rect = sp['rect']            # keep reference, do not replace
            self.initial_rect = self.rect.copy()

            # Physics state
            self.pos = pygame.Vector2(self.rect.topleft)  # float position (topleft)
            self.vel = pygame.Vector2(0, 0)

            # Tunable physics parameters
            self.gravity_on = True
            self.gravity = 1400.0           # px/s^2
            self.restitution = 0.78         # bounciness
            self.air_drag = 0.14            # per-second air resistance
            self.floor_friction = 0.9       # applied on floor hits
            self.sleep_threshold = 28.0     # speed threshold to stop tiny bounces

            # Interaction
            self.dragging = False
            self.drag_offset_center = pygame.Vector2(0, 0)
            self.drag_history = []          # list of (time_sec, center_pos)
            self.paused = False

            # Audio helpers (optional)
            audio = context.get("audio", {})
            self.SAMPLE_RATE = audio.get("SAMPLE_RATE", None)
            self.make_sound_from_wave = audio.get("make_sound_from_wave", None)
            self.last_bounce_time = 0  # ms

        # ----- Utility/audio -----
        def _now(self):
            return pygame.time.get_ticks() / 1000.0

        def _play_bounce(self, impact_speed):
            if not (self.SAMPLE_RATE and self.make_sound_from_wave):
                return
            # Rate-limit bounces a bit
            now_ms = pygame.time.get_ticks()
            if now_ms - self.last_bounce_time < 50:
                return
            self.last_bounce_time = now_ms

            # Short click with freq based on impact
            dur = 0.045
            speed = max(0.0, min(impact_speed, 2200.0))
            freq = 180 + 520 * (speed / 2200.0)
            t = np.linspace(0, dur, int(self.SAMPLE_RATE * dur), endpoint=False)
            wave = 0.6 * np.sin(2 * np.pi * freq * t) * np.linspace(1, 0.9, t.size)
            snd = self.make_sound_from_wave(wave, volume=0.7)
            snd.play()

        def _update_rect_from_pos(self):
            self.rect.topleft = (int(round(self.pos.x)), int(round(self.pos.y)))

        # ----- Event handling -----
        def handle_event(self, event):
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    self.gravity_on = not self.gravity_on
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._reset()
                else:
                    # Arrow key nudges
                    nudge = 8
                    if event.key == pygame.K_LEFT:
                        self.pos.x -= nudge
                    elif event.key == pygame.K_RIGHT:
                        self.pos.x += nudge
                    elif event.key == pygame.K_UP:
                        self.pos.y -= nudge
                    elif event.key == pygame.K_DOWN:
                        self.pos.y += nudge
                    self._update_rect_from_pos()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.rect.collidepoint(event.pos):
                    self.dragging = True
                    mouse = pygame.Vector2(event.pos)
                    self.drag_offset_center = mouse - pygame.Vector2(self.rect.center)
                    self.drag_history.clear()
                    self.vel.update(0, 0)  # while dragging, stop motion
                    # seed history
                    self._record_drag_point(mouse)

            elif event.type == pygame.MOUSEMOTION and self.dragging:
                mouse = pygame.Vector2(event.pos)
                center = mouse - self.drag_offset_center
                self.rect.center = (int(round(center.x)), int(round(center.y)))
                self.pos.update(self.rect.topleft)
                self._record_drag_point(mouse)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
                self.dragging = False
                # Compute throw velocity from recent pointer movement
                self.vel = self._compute_throw_velocity()
                self.drag_history.clear()

        def _record_drag_point(self, mouse_pos):
            now = self._now()
            # store center position
            center = mouse_pos - self.drag_offset_center
            self.drag_history.append((now, pygame.Vector2(center)))
            # keep only last ~150 ms
            cutoff = now - 0.15
            while len(self.drag_history) > 1 and self.drag_history[0][0] < cutoff:
                self.drag_history.pop(0)

        def _compute_throw_velocity(self):
            if len(self.drag_history) < 2:
                return pygame.Vector2(0, 0)
            t1, p1 = self.drag_history[-1]
            # find an earlier point at least 20 ms before last
            idx = len(self.drag_history) - 2
            while idx > 0 and t1 - self.drag_history[idx][0] < 0.02:
                idx -= 1
            t0, p0 = self.drag_history[idx]
            dt = max(1e-4, t1 - t0)
            v_center = (p1 - p0) / dt
            # Convert center-velocity to topleft-velocity (same)
            # Scale down to keep throws reasonable
            v = v_center * 0.85
            # Clamp speed
            max_speed = 1900.0
            if v.length() > max_speed:
                v.scale_to_length(max_speed)
            return v

        def _reset(self):
            self.rect.update(self.initial_rect)  # copies position/size
            self.pos.update(self.rect.topleft)
            self.vel.update(0, 0)
            self.paused = False

        # ----- Simulation -----
        def update(self, dt):
            # Clamp dt to avoid tunneling on frame hiccups
            dt = max(0.0, min(dt, 0.05))
            if self.paused:
                return

            if self.dragging:
                # While dragging, physics halted; position already set in events
                self.vel.update(0, 0)
                return

            # Gravity
            if self.gravity_on:
                self.vel.y += self.gravity * dt

            # Air drag (exponential)
            drag_factor = math.exp(-self.air_drag * dt)
            self.vel *= drag_factor

            # Integrate
            self.pos += self.vel * dt
            self._update_rect_from_pos()

            # Collisions with window edges
            w, h = self.screen.get_size()
            collided = False
            impact_speed = 0.0

            # Left
            if self.rect.left < 0:
                self.rect.left = 0
                self.pos.x = self.rect.x
                impact_speed = max(impact_speed, abs(self.vel.x))
                self.vel.x = -self.vel.x * self.restitution
                collided = True
            # Right
            if self.rect.right > w:
                self.rect.right = w
                self.pos.x = self.rect.x
                impact_speed = max(impact_speed, abs(self.vel.x))
                self.vel.x = -self.vel.x * self.restitution
                collided = True
            # Top
            if self.rect.top < 0:
                self.rect.top = 0
                self.pos.y = self.rect.y
                impact_speed = max(impact_speed, abs(self.vel.y))
                self.vel.y = -self.vel.y * self.restitution
                collided = True
            # Bottom
            if self.rect.bottom > h:
                self.rect.bottom = h
                self.pos.y = self.rect.y
                impact_speed = max(impact_speed, abs(self.vel.y))
                # Bounce only if falling; otherwise just clamp
                if self.vel.y > 0:
                    self.vel.y = -self.vel.y * self.restitution
                # Floor friction and sleep when slow
                if abs(self.vel.y) < self.sleep_threshold:
                    self.vel.y = 0.0
                self.vel.x *= self.floor_friction
                collided = True

            if collided and impact_speed > 60:
                self._play_bounce(impact_speed)

        # ----- Rendering -----
        def draw(self, screen):
            # Background already drawn by the shell
            screen.blit(self.surf, self.rect)

        # Provide object-like interface explicitly
    return BouncyBallGame(screen, background_surf, context)
