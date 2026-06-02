"""
Flood organism - the adversary that creates selection pressure.
Flood have their own simple AI: seek, infect, reproduce.
They are attracted to Lekgolo colonies and try to infect them,
with a preference for thinkers.
"""
import numpy as np
from config import (
    FLOOD_HEALTH, FLOOD_STRENGTH, FLOOD_SPEED, FLOOD_ATTACK_DAMAGE,
    FLOOD_INFECTION_DAMAGE, FLOOD_ATTACK_RANGE, FLOOD_VISION,
    FLOOD_REPRODUCE_INTERVAL, FLOOD_REPRODUCE_BIOMASS_COST,
    MAP_WIDTH, MAP_HEIGHT
)


class FloodOrganism:
    """A single Flood organism."""

    _next_id = 0

    def __init__(self, x: float, y: float):
        self.id = FloodOrganism._next_id
        FloodOrganism._next_id += 1

        self.x = x
        self.y = y
        self.health = FLOOD_HEALTH
        self.max_health = FLOOD_HEALTH
        self.strength = FLOOD_STRENGTH
        self.alive = True
        self.orientation = np.random.uniform(0, 2 * np.pi)
        self.reproduce_timer = FLOOD_REPRODUCE_INTERVAL
        self.biomass = FLOOD_REPRODUCE_BIOMASS_COST  # starts ready to reproduce

    def take_damage(self, amount: float) -> float:
        if not self.alive:
            return 0.0
        actual = min(amount, self.health)
        self.health -= actual
        if self.health <= 0:
            self.health = 0
            self.alive = False
        return actual

    def distance_to(self, other_x: float, other_y: float) -> float:
        return np.sqrt((self.x - other_x) ** 2 + (self.y - other_y) ** 2)

    def choose_action(self, nearby_worms: list, nearby_flood: list,
                      terrain: np.ndarray) -> dict:
        """
        Simple Flood AI: move toward nearest worm (preferring thinkers),
        attack/infect if in range, reproduce if ready.

        Returns an action dict.
        """
        if not self.alive:
            return {'type': 'idle'}

        # Find nearest worm, preferring thinkers
        target = None
        target_dist = float('inf')

        for worm in nearby_worms:
            if not worm.alive:
                continue
            d = self.distance_to(worm.x, worm.y)
            # Thinker preference: effectively reduce distance by 30%
            if worm.worm_type == 1:  # THINKER
                d *= 0.7
            if d < target_dist:
                target_dist = d
                target = worm

        action = {'type': 'move', 'dx': 0.0, 'dy': 0.0}

        if target is not None and target_dist <= FLOOD_ATTACK_RANGE:
            # Attack/infect
            is_thinker = target.worm_type == 1
            action = {
                'type': 'attack',
                'target_id': target.id,
                'is_thinker': is_thinker,
            }
        elif target is not None:
            # Move toward target
            dx = target.x - self.x
            dy = target.y - self.y
            dist = max(np.sqrt(dx * dx + dy * dy), 1e-6)
            speed = FLOOD_SPEED
            action = {
                'type': 'move',
                'dx': (dx / dist) * speed,
                'dy': (dy / dist) * speed,
            }
        else:
            # Wander randomly
            angle = np.random.uniform(0, 2 * np.pi)
            action = {
                'type': 'move',
                'dx': np.cos(angle) * FLOOD_SPEED,
                'dy': np.sin(angle) * FLOOD_SPEED,
            }

        # Reproduce if timer is ready
        if self.reproduce_timer <= 0 and self.biomass >= FLOOD_REPRODUCE_BIOMASS_COST:
            action['reproduce'] = True
        else:
            action['reproduce'] = False

        return action

    def tick_timers(self):
        if self.alive:
            self.reproduce_timer = max(0, self.reproduce_timer - 1)

    def can_reproduce(self) -> bool:
        return (self.alive and
                self.reproduce_timer <= 0 and
                self.biomass >= FLOOD_REPRODUCE_BIOMASS_COST)
