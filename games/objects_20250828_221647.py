import pygame
import numpy as np

def create_game(screen, background_surf, context):
    class PotAndWaterGame:
        def __init__(self, screen, context):
            self.screen = screen
            self.w, self.h = self.screen.get_size()
            self.audio = context.get("audio", {})
            self.SR = self.audio.get("SAMPLE_RATE", 44100)
            self.make_sound_from_wave = self.audio.get("make_sound_from_wave", None)

            # Approximate regions based on the provided drawing layout
            # Soil ellipse (pot top) — left-middle area
            self.soil_rect = pygame.Rect(
                int(self.w * 0.26), int(self.h * 0.56),
                int(self.w * 0.20), int(self.h * 0.07)
            )
            # Watering can body and spout — right side area
            self.can_rect = pygame.Rect(
                int(self.w * 0.62), int(self.h * 0.40),
                int(self.w * 0.22), int(self.h * 0.33)
            )
            self.spout_rect = pygame.Rect(
                int(self.w * 0.56), int(self.h * 0.46),
                int(self.w * 0.07), int(self.h * 0.10)
            )

            # Spout tip (droplet origin)
            self.spout_tip = (
                self.spout_rect.left + int(self.spout_rect.width * 0.15),
                self.spout_rect.centery
            )

            # Plant growth state
            self.base_x = self.soil_rect.centerx
            self.base_y = self.soil_rect.centery
            self.plant_height = 0.0
            self.target_height = 0.0
            self.max_height = self.h * 0.45
            self.growth_speed = max(60, int(self.h * 0.12))  # px/sec

            # Watering animation
            self.watering_timer = 0.0
            self.watering_duration = 1.2
            self.spawn_rate = 80.0  # droplets per second
            self.spawn_accum = 0.0
            self.droplets = []
            self.gravity = self.h * 1.8  # px/sec^2

            # Hover flags
            self.hover_can = False
            self.hover_pot = False

            # Visuals
            self.green = (40, 160, 60)
            self.dark_green = (20, 120, 45)
            self.blue = (70, 140, 220)

        # ----------------- Audio -----------------
        def play_water(self, dur=1.0):
            if not (self.make_sound_from_wave and self.SR):
                return
            t = np.linspace(0.0, dur, int(self.SR * dur), endpoint=False)
            # Water-like noise: white noise with a gentle envelope and small ripples
            noise = np.random.normal(0.0, 1.0, t.shape)
            # Amplitude modulation to emulate splashes/dribble
            am = 0.5 + 0.5 * np.sin(2 * np.pi * 5.5 * t + 2 * np.pi * np.random.rand())
            am += 0.3 * np.sin(2 * np.pi * 12.0 * t + 2 * np.pi * np.random.rand())
            am = np.clip(am, 0.0, 1.0)

            # Overall envelope with quick attack, slow decay
            env = np.exp(-3.0 * t)
            # Fade in/out to avoid clicks
            fade_len = max(1, int(0.01 * self.SR))
            fade_in = np.linspace(0.0, 1.0, fade_len)
            fade_out = fade_in[::-1]
            env[:fade_len] *= fade_in
            env[-fade_len:] *= fade_out

            wave = 0.7 * noise * env * am
            # Add subtle high-frequency sparkle
            wave += 0.08 * np.sin(2 * np.pi * (700 + 300*np.sin(2*np.pi*1.5*t)) * t) * env

            # Normalize to safe range
            mx = np.max(np.abs(wave)) + 1e-6
            wave = 0.9 * (wave / mx)

            snd = self.make_sound_from_wave(wave, volume=0.8)
            snd.play()

        # ----------------- Interaction helpers -----------------
        def _in_can(self, pos):
            return self.can_rect.collidepoint(pos) or self.spout_rect.collidepoint(pos)

        def _in_pot(self, pos):
            # Elliptical hit test for soil top
            rx = self.soil_rect.width / 2.0
            ry = self.soil_rect.height / 2.0
            cx, cy = self.soil_rect.center
            x, y = pos
            if rx <= 0 or ry <= 0:
                return False
            nx = (x - cx) / rx
            ny = (y - cy) / ry
            return nx * nx + ny * ny <= 1.0

        def start_watering(self):
            self.watering_timer = self.watering_duration
            self.spawn_accum = 0.0
            # Kick plant target upward; multiple clicks stack but clamp
            self.target_height = min(self.max_height, self.target_height + self.h * 0.12)
            self.play_water(self.watering_duration)

        # ----------------- Pygame interface -----------------
        def handle_event(self, event):
            if event.type == pygame.MOUSEMOTION:
                pos = event.pos
                self.hover_can = self._in_can(pos)
                self.hover_pot = self._in_pot(pos)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if self._in_can(pos):
                    self.start_watering()
                elif self._in_pot(pos):
                    # Gentle trim/tidy: slight sway boost
                    self.target_height = min(self.max_height, self.target_height + self.h * 0.04)

        def update(self, dt):
            dt = max(0.0, min(0.05, dt))  # clamp for stability

            # Plant growth towards target
            if self.plant_height < self.target_height:
                self.plant_height = min(self.target_height, self.plant_height + self.growth_speed * dt)
            elif self.plant_height > self.target_height:
                self.plant_height = max(self.target_height, self.plant_height - (self.growth_speed * 0.6) * dt)

            # Watering: spawn droplets
            if self.watering_timer > 0.0:
                self.spawn_accum += self.spawn_rate * dt
                while self.spawn_accum >= 1.0:
                    self.spawn_accum -= 1.0
                    self._spawn_droplet()
                self.watering_timer -= dt

            # Update droplets physics
            alive = []
            soil_y = self.base_y + self.soil_rect.height * 0.1
            left_bound = self.soil_rect.left - self.w * 0.05
            right_bound = self.soil_rect.right + self.w * 0.05
            for d in self.droplets:
                d["vy"] += self.gravity * dt
                d["x"] += d["vx"] * dt
                d["y"] += d["vy"] * dt
                d["life"] -= dt
                # Remove if out or hit soil area ellipse
                if d["life"] <= 0:
                    continue
                if left_bound <= d["x"] <= right_bound and d["y"] >= soil_y:
                    # Consider it absorbed by soil
                    continue
                if d["x"] < -self.w * 0.1 or d["x"] > self.w * 1.1 or d["y"] > self.h * 1.1:
                    continue
                alive.append(d)
            self.droplets = alive

        def _spawn_droplet(self):
            # Aim generally toward the pot center with some randomness
            sx, sy = self.spout_tip
            tx = self.base_x + (np.random.rand() - 0.5) * self.soil_rect.width * 0.4
            ty = self.base_y + (np.random.rand() - 0.3) * self.soil_rect.height * 0.4

            # Initial velocity to arc from spout to soil
            # Choose a flight time and solve approximately
            t_flight = 0.7 + 0.25 * (np.random.rand() - 0.5)
            t_flight = max(0.45, t_flight)
            vx = (tx - sx) / t_flight
            vy = (ty - sy - 0.5 * self.gravity * t_flight * t_flight) / t_flight

            self.droplets.append({
                "x": float(sx),
                "y": float(sy),
                "vx": float(vx),
                "vy": float(vy),
                "r": max(2, int(self.h * (0.004 + 0.003 * np.random.rand()))),
                "life": t_flight * 1.2
            })

        def draw(self, screen):
            # Optional hover highlights to indicate interactivity
            self._draw_highlights(screen)
            # Draw dynamic plant
            self._draw_plant(screen)
            # Draw water droplets
            self._draw_droplets(screen)

        # ----------------- Rendering helpers -----------------
        def _draw_highlights(self, screen):
            if self.hover_can:
                s = pygame.Surface((self.can_rect.width, self.can_rect.height), pygame.SRCALPHA)
                s.fill((50, 120, 255, 40))
                screen.blit(s, self.can_rect.topleft)
                s2 = pygame.Surface((self.spout_rect.width, self.spout_rect.height), pygame.SRCALPHA)
                s2.fill((50, 120, 255, 40))
                screen.blit(s2, self.spout_rect.topleft)
            if self.hover_pot:
                s = pygame.Surface(self.soil_rect.size, pygame.SRCALPHA)
                s.fill((60, 200, 80, 40))
                screen.blit(s, self.soil_rect.topleft)

        def _draw_plant(self, screen):
            if self.plant_height <= 1:
                return
            h = self.plant_height
            # Build a gently curving stem polyline
            segments = max(10, int(h / 8))
            xs = []
            ys = []
            sway_amp = max(6, int(self.w * 0.01))
            phase = pygame.time.get_ticks() * 0.0015
            for i in range(segments + 1):
                t = i / segments
                y = self.base_y - h * t
                # curve that leans slightly toward the can, and gently sways
                lean = -0.08 * (1 - np.cos(np.pi * t)) * self.soil_rect.width
                sway = sway_amp * np.sin(phase + 3.0 * t) * t
                x = self.base_x + lean + sway
                xs.append(int(x))
                ys.append(int(y))

            # Draw stem
            for i in range(1, len(xs)):
                thickness = max(2, int(3 + 2 * (i / len(xs))))
                pygame.draw.line(screen, self.green, (xs[i-1], ys[i-1]), (xs[i], ys[i]), thickness)

            # Draw leaves along the stem
            leaves = max(1, int(h / 30))
            for j in range(leaves):
                t = (j + 1) / (leaves + 1)
                # Interpolate along polyline
                idx = min(int(t * segments), segments)
                cx, cy = xs[idx], ys[idx]
                side = -1 if j % 2 == 0 else 1
                leaf_len = int(12 + 10 * (j / max(1, leaves - 1)))
                leaf_w = int(6 + 4 * (j / max(1, leaves - 1)))
                self._draw_leaf(screen, cx, cy, side, leaf_len, leaf_w)

            # Bud/flower at top
            pygame.draw.circle(screen, self.dark_green, (xs[-1], ys[-1]), 5)

        def _draw_leaf(self, screen, cx, cy, side, leaf_len, leaf_w):
            # Simple tapered polygon leaf
            tip = (cx + side * leaf_len, cy)
            up = (cx + side * int(leaf_len * 0.6), cy - leaf_w)
            dn = (cx + side * int(leaf_len * 0.6), cy + leaf_w)
            pygame.draw.polygon(screen, self.green, [ (cx, cy), up, tip, dn ])

        def _draw_droplets(self, screen):
            for d in self.droplets:
                pygame.draw.circle(screen, self.blue, (int(d["x"]), int(d["y"])), d["r"])

    return PotAndWaterGame(screen, context)
