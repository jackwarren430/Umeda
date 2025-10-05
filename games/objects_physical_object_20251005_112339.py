import math
import pygame

try:
    import numpy as np
except Exception:
    np = None  # Audio will be disabled if numpy is unavailable


def create_game(screen, background_surf, context):
    '''
    Returns an object with methods:
      handle_event(event)    # called for each event
      update(dt)             # dt: float seconds
      draw(screen)           # draw your overlays/sprites; background is already drawn
    '''
    class BouncyBallGame:
        def __init__(self, screen, context):
            assert "sprite" in context and isinstance(context["sprite"], dict), "sprite missing from context"
            spr = context["sprite"]
            self.surf = spr["surface"]
            self.rect = spr["rect"].copy()
            # Physics state (use float position)
            self.pos = pygame.Vector2(self.rect.topleft)
            self.vel = pygame.Vector2(0.0, 0.0)

            # Physics parameters
            self.gravity_enabled = True
            self.g = 1800.0  # px/s^2
            self.linear_drag = 0.15  # per-second damping
            self.restitution = 0.85  # bounce energy retention
            self.floor_stop_threshold = 20.0  # px/s to settle

            # Controls / state
            self.paused = False
            self.dragging = False
            self.drag_offset = pygame.Vector2(0, 0)
            self.mouse_samples = []  # (time, (x, y)) for throw velocity
            self.sample_window = 0.14
            self.max_throw_speed = 2400.0
            self.throw_gain = 1.0

            # Keep initial state for reset
            self.initial_pos = self.pos.copy()
            self.initial_vel = self.vel.copy()
            self.initial_gravity = self.gravity_enabled

            # Audio helpers (optional)
            self.time = 0.0
            self.last_bounce_time = -1e9
            self.audio_ok = False
            self.SAMPLE_RATE = None
            self.make_sound_from_wave = None
            if "audio" in context and np is not None:
                audio = context["audio"]
                self.SAMPLE_RATE = audio.get("SAMPLE_RATE", None)
                self.make_sound_from_wave = audio.get("make_sound_from_wave", None)
                self.audio_ok = self.SAMPLE_RATE is not None and callable(self.make_sound_from_wave)

        # ---------- Controls ----------
        def handle_event(self, event):
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.rect.collidepoint(event.pos):
                    self.dragging = True
                    self.drag_offset = pygame.Vector2(event.pos) - pygame.Vector2(self.rect.topleft)
                    self.mouse_samples.clear()
                    self._record_mouse_sample(event.pos)
                    # Zero velocity while dragging to avoid fighting with input
                    self.vel.update(0.0, 0.0)

            elif event.type == pygame.MOUSEMOTION and self.dragging:
                new_top_left = pygame.Vector2(event.pos) - self.drag_offset
                self.pos.update(new_top_left.x, new_top_left.y)
                self.rect.topleft = (int(self.pos.x), int(self.pos.y))
                self._record_mouse_sample(event.pos)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
                self.dragging = False
                self._record_mouse_sample(event.pos)
                throw_v = self._compute_throw_velocity()
                if throw_v is not None:
                    self.vel = throw_v

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    self.gravity_enabled = not self.gravity_enabled
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._reset()
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
                    step = 10
                    if event.mod & pygame.KMOD_SHIFT:
                        step = 30
                    if event.key == pygame.K_LEFT:
                        self.pos.x -= step
                    elif event.key == pygame.K_RIGHT:
                        self.pos.x += step
                    elif event.key == pygame.K_UP:
                        self.pos.y -= step
                    elif event.key == pygame.K_DOWN:
                        self.pos.y += step
                    self.rect.topleft = (int(self.pos.x), int(self.pos.y))

        # ---------- Simulation ----------
        def update(self, dt):
            self.time += dt
            if self.paused or self.dragging:
                return

            # Apply gravity and drag, then integrate position (semi-implicit Euler)
            if self.gravity_enabled:
                self.vel.y += self.g * dt

            # Linear air drag as exponential damping
            if self.linear_drag > 0:
                damp = math.exp(-self.linear_drag * dt)
                self.vel.x *= damp
                self.vel.y *= damp

            self.pos += self.vel * dt
            self.rect.topleft = (int(self.pos.x), int(self.pos.y))

            # Collisions with window edges
            w, h = screen.get_size()
            collided = False
            speed_before = self.vel.length()

            # Left
            if self.rect.left < 0:
                self.rect.left = 0
                self.pos.x = float(self.rect.left)
                self.vel.x = -self.vel.x * self.restitution
                collided = True
            # Right
            if self.rect.right > w:
                self.rect.right = w
                self.pos.x = float(self.rect.left)
                self.vel.x = -self.vel.x * self.restitution
                collided = True
            # Top
            if self.rect.top < 0:
                self.rect.top = 0
                self.pos.y = float(self.rect.top)
                self.vel.y = -self.vel.y * self.restitution
                collided = True
            # Bottom
            if self.rect.bottom > h:
                self.rect.bottom = h
                self.pos.y = float(self.rect.top)
                self.vel.y = -self.vel.y * self.restitution
                # Simulate a little floor friction on bounce
                self.vel.x *= 0.98
                # If very slow, settle
                if abs(self.vel.y) < self.floor_stop_threshold and self.gravity_enabled:
                    self.vel.y = 0.0
                collided = True

            # Play bounce sound if strong enough and not too frequent
            if collided:
                speed_after = self.vel.length()
                impact_speed = max(speed_before, speed_after)
                if impact_speed > 120.0:
                    self._play_bounce(impact_speed)

        # ---------- Rendering ----------
        def draw(self, screen):
            screen.blit(self.surf, self.rect)

        # ---------- Helpers ----------
        def _record_mouse_sample(self, pos):
            now = self.time
            self.mouse_samples.append((now, (float(pos[0]), float(pos[1]))))
            # Drop old samples
            cutoff = now - self.sample_window
            while len(self.mouse_samples) > 2 and self.mouse_samples[0][0] < cutoff:
                self.mouse_samples.pop(0)

        def _compute_throw_velocity(self):
            if len(self.mouse_samples) < 2:
                return None
            t_end, p_end = self.mouse_samples[-1]
            # Find earliest sample within window
            idx = 0
            for i in range(len(self.mouse_samples) - 2, -1, -1):
                if t_end - self.mouse_samples[i][0] > self.sample_window:
                    idx = i + 1
                    break
                idx = 0
            t_start, p_start = self.mouse_samples[idx]
            dt = max(1e-4, t_end - t_start)
            vx = (p_end[0] - p_start[0]) / dt
            vy = (p_end[1] - p_start[1]) / dt
            v = pygame.Vector2(vx, vy) * self.throw_gain
            # Clamp throw speed
            speed = v.length()
            if speed > self.max_throw_speed:
                v.scale_to_length(self.max_throw_speed)
            return v

        def _reset(self):
            self.pos = self.initial_pos.copy()
            self.vel = self.initial_vel.copy()
            self.gravity_enabled = self.initial_gravity
            self.rect.topleft = (int(self.pos.x), int(self.pos.y))

        def _play_bounce(self, speed):
            if not self.audio_ok or self.time - self.last_bounce_time < 0.05:
                return
            self.last_bounce_time = self.time
            # Map speed to pitch and duration
            s = max(0.0, min(speed, 1800.0)) / 1800.0
            freq = 250.0 + 900.0 * s
            dur = 0.03 + 0.05 * s
            t = np.linspace(0, dur, int(self.SAMPLE_RATE * dur), endpoint=False)
            # Simple sine with tiny fade
            wave = 0.6 * np.sin(2 * np.pi * freq * t) * np.linspace(1.0, 0.9, t.size)
            try:
                snd = self.make_sound_from_wave(wave, volume=0.8)
                snd.play()
            except Exception:
                # If audio fails, just ignore
                pass

    return BouncyBallGame(screen, context)
