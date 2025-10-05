import math
import random
import pygame
import numpy as np


def create_game(screen, background_surf, context):
    '''
    Returns an object with methods:
      handle_event(event)
      update(dt)
      draw(screen)
    '''

    class ParticleEmitterGame:
        def __init__(self, screen, background_surf, context):
            self.screen_size = screen.get_size()
            self.context = context or {}

            # Emitter parameters
            self.emitting = False
            self.gravity_on = True
            self.gravity = 900.0  # px/s^2
            self.drag = 1.6       # linear drag factor per second
            self.base_speed = 650.0
            self.spread_deg = 18.0  # cone half-width
            self.rate = 320.0       # particles per second
            self.dir_angle = 0.0    # aim to the right by default

            # Emitter origin
            self.init_origin = (self.screen_size[0] // 2, self.screen_size[1] // 2)
            self.origin = list(self.init_origin)

            # Build particle artwork once
            self.base_particle = self._make_particle_surface(context)
            # Precompute multiple scales (no surface creation during draw)
            self.scale_levels = [0.45 + 0.05 * i for i in range(16)]  # 0.45 .. 1.2
            self.scaled_surfs = self._precompute_scaled(self.base_particle, self.scale_levels)

            # Pool
            self.MAX_PARTICLES = 1200
            self._init_pool(self.MAX_PARTICLES)

            # Emission accumulator
            self._emit_acc = 0.0

            # Optional short click sound on burst
            self._burst_sound = self._make_burst_sound(self.context.get("audio"))

            # If a sprite is provided, set origin to its center
            sprite = context.get("sprite")
            if sprite and "rect" in sprite:
                cx, cy = sprite["rect"].center
                self.origin = [float(cx), float(cy)]
                self.init_origin = (float(cx), float(cy))

        # ----------------------- Setup helpers -----------------------
        def _make_particle_surface(self, context):
            # If a sprite exists, derive a small, alpha-preserving particle from it
            sprite = context.get("sprite")
            if sprite and "surface" in sprite and isinstance(sprite["surface"], pygame.Surface):
                srf = sprite["surface"]
                w, h = srf.get_size()
                # Scale so the max dimension ~24 px
                max_dim = max(w, h)
                scale = 24.0 / max(1, max_dim)
                new_size = max(2, int(w * scale)), max(2, int(h * scale))
                img = pygame.transform.smoothscale(srf, new_size)
                # Tint a bit towards water-blue while keeping alpha
                tint = pygame.Surface(img.get_size(), pygame.SRCALPHA)
                tint.fill((60, 140, 255, 255))
                img = img.copy()
                img.blit(tint, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                return img

            # Otherwise synthesize a soft round droplet (18-22 px)
            size = 22
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            # Radial soft alpha, blue color
            cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
            yy, xx = np.mgrid[0:size, 0:size]
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            r /= (size / 2.0)
            alpha = np.clip(1.0 - r, 0.0, 1.0)
            alpha = (alpha ** 1.5)  # softer edge
            color = np.zeros((size, size, 4), dtype=np.uint8)
            color[..., 0] = 70
            color[..., 1] = 170
            color[..., 2] = 255
            color[..., 3] = (alpha * 255).astype(np.uint8)
            pygame.surfarray.blit_array(surf, color.swapaxes(0, 1))
            return surf

        def _precompute_scaled(self, base, scales):
            bw, bh = base.get_size()
            images = []
            for s in scales:
                sw = max(1, int(bw * s))
                sh = max(1, int(bh * s))
                images.append(pygame.transform.smoothscale(base, (sw, sh)))
            return images

        def _init_pool(self, capacity):
            self.active = []                 # list of indices of active particles
            self.free = list(range(capacity))  # freelist
            # Arrays
            self.px = [0.0] * capacity
            self.py = [0.0] * capacity
            self.vx = [0.0] * capacity
            self.vy = [0.0] * capacity
            self.life = [0.0] * capacity
            self.life_max = [1.0] * capacity
            self.s0 = [0] * capacity   # scale index start
            self.s1 = [0] * capacity   # scale index end (for over-lifetime change)

        def _make_burst_sound(self, audio):
            try:
                if not audio:
                    return None
                SAMPLE_RATE = audio["SAMPLE_RATE"]
                make_sound_from_wave = audio["make_sound_from_wave"]
                dur = 0.06
                t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
                # short "squirt" noise burst with slight highpass character
                noise = np.random.uniform(-1.0, 1.0, t.size)
                env = np.exp(-t * 40.0)
                wave = 0.35 * noise * env
                snd = make_sound_from_wave(wave, volume=0.7)
                return snd
            except Exception:
                return None

        # ----------------------- Pool ops -----------------------
        def _spawn_particle(self, x, y, vx, vy, life, sidx0, sidx1):
            if not self.free:
                # If full, recycle oldest
                idx = self.active.pop(0)
            else:
                idx = self.free.pop()
            self.px[idx] = x
            self.py[idx] = y
            self.vx[idx] = vx
            self.vy[idx] = vy
            self.life[idx] = life
            self.life_max[idx] = life
            self.s0[idx] = sidx0
            self.s1[idx] = sidx1
            self.active.append(idx)

        def _burst(self, x, y, count=60):
            # Create a burst pointing roughly along dir_angle with spread
            half = math.radians(self.spread_deg)
            for _ in range(count):
                ang = self.dir_angle + random.uniform(-half, half)
                spd = self.base_speed * random.uniform(0.8, 1.15)
                vx = math.cos(ang) * spd
                vy = math.sin(ang) * spd
                life = random.uniform(0.35, 0.75)
                # size over life: small shrink
                s0 = random.randrange(6, len(self.scale_levels) - 1)
                s1 = max(0, s0 - random.randint(1, 5))
                self._spawn_particle(x, y, vx, vy, life, s0, s1)
            if self._burst_sound:
                try:
                    self._burst_sound.play()
                except Exception:
                    pass

        # ----------------------- Controls -----------------------
        def handle_event(self, event):
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                self.origin[0], self.origin[1] = float(mx), float(my)
                # Aim towards the cursor direction vs previous origin position
                self.dir_angle = 0.0  # keep rightward look; consistent stream
                self._burst(self.origin[0], self.origin[1], count=70)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.emitting = not self.emitting
                elif event.key == pygame.K_g:
                    self.gravity_on = not self.gravity_on
                elif event.key == pygame.K_r:
                    # reset particles and origin
                    self._init_pool(self.MAX_PARTICLES)
                    self.origin = [float(self.init_origin[0]), float(self.init_origin[1])]
                elif event.key == pygame.K_LEFT:
                    self.spread_deg = max(2.0, self.spread_deg - 3.0)
                elif event.key == pygame.K_RIGHT:
                    self.spread_deg = min(80.0, self.spread_deg + 3.0)
                elif event.key == pygame.K_UP:
                    self.base_speed = max(80.0, self.base_speed - 40.0)
                elif event.key == pygame.K_DOWN:
                    self.base_speed = min(1600.0, self.base_speed + 40.0)
                elif event.key == pygame.K_LEFTBRACKET:
                    self.rate = max(10.0, self.rate - 40.0)
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.rate = min(2000.0, self.rate + 40.0)

        # ----------------------- Update -----------------------
        def update(self, dt):
            # Continuous emission from origin
            if self.emitting and self.rate > 0:
                self._emit_acc += self.rate * dt
                n = int(self._emit_acc)
                if n > 0:
                    self._emit_acc -= n
                    # Slight jitter around the origin to look like water stream aperture
                    for _ in range(n):
                        ang = self.dir_angle + random.uniform(
                            -math.radians(self.spread_deg), math.radians(self.spread_deg)
                        )
                        spd = self.base_speed * random.uniform(0.9, 1.1)
                        vx = math.cos(ang) * spd
                        vy = math.sin(ang) * spd
                        life = random.uniform(0.25, 0.55)
                        s0 = random.randrange(5, len(self.scale_levels) - 1)
                        s1 = max(0, s0 - random.randint(1, 4))
                        ox = self.origin[0] + random.uniform(-1.5, 1.5)
                        oy = self.origin[1] + random.uniform(-1.5, 1.5)
                        self._spawn_particle(ox, oy, vx, vy, life, s0, s1)

            # Physics update for active particles
            g = self.gravity if self.gravity_on else 0.0
            drag = self.drag
            # Iterate backwards so we can pop safely on death
            for i in range(len(self.active) - 1, -1, -1):
                idx = self.active[i]
                # Life
                self.life[idx] -= dt
                if self.life[idx] <= 0:
                    self.free.append(idx)
                    self.active.pop(i)
                    continue
                # Motion
                if g != 0.0:
                    self.vy[idx] += g * dt
                # Linear drag
                d = max(0.0, 1.0 - drag * dt)
                self.vx[idx] *= d
                self.vy[idx] *= d
                self.px[idx] += self.vx[idx] * dt
                self.py[idx] += self.vy[idx] * dt

        # ----------------------- Draw -----------------------
        def draw(self, screen):
            # Draw particles on top of the already-blitted background
            for idx in self.active:
                # Alpha fades with remaining life
                t = 1.0 - (self.life[idx] / self.life_max[idx])
                # Scale interpolate from s0 to s1 over life
                sidx = int(self.s0[idx] * (1.0 - t) + self.s1[idx] * t + 0.5)
                sidx = max(0, min(sidx, len(self.scaled_surfs) - 1))
                img = self.scaled_surfs[sidx]
                a = int(255 * (1.0 - t))
                if a <= 0:
                    continue
                img.set_alpha(a)
                rect = img.get_rect(center=(int(self.px[idx]), int(self.py[idx])))
                screen.blit(img, rect)

            # Optional: small nozzle indicator when emitting
            if self.emitting:
                pygame.draw.circle(screen, (80, 160, 255), (int(self.origin[0]), int(self.origin[1])), 2)

    return ParticleEmitterGame(screen, background_surf, context)
