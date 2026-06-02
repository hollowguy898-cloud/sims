"""
Lekgolo Worm agent.
Each worm is an individual agent with its own state, sensors, and policy.
Worm types (Worker / Thinker) differ in capacity, not in architecture.
"""
import math
import enum
from config import (
    WORM_MAX_HEALTH_WORKER, WORM_MAX_HEALTH_THINKER,
    WORM_STRENGTH_WORKER, WORM_STRENGTH_THINKER,
    WORM_VISION_RADIUS_WORKER, WORM_VISION_RADIUS_THINKER,
    WORM_MAX_ENERGY, WORM_ATTACK_DAMAGE_WORKER, WORM_ATTACK_DAMAGE_THINKER,
    WORM_ATTACK_RANGE, SIGNAL_DIM, COMM_RADIUS_WORKER, COMM_RADIUS_THINKER,
    MAX_ATTACHMENTS_PER_WORM, WORKER_HIDDEN_DIM, THINKER_HIDDEN_DIM,
    MAP_WIDTH, MAP_HEIGHT
)


class WormType(enum.IntEnum):
    WORKER = 0
    THINKER = 1


class Worm:
    """A single Lekgolo worm agent."""

    _next_id = 0

    def __init__(self, x: float, y: float, worm_type: WormType = WormType.WORKER,
                 rng=None):
        self.id = Worm._next_id
        Worm._next_id += 1

        self.x = x
        self.y = y
        self.worm_type = worm_type

        # Orientation as angle in radians
        if rng is not None:
            self.orientation = rng.uniform(0, 2 * math.pi)
        else:
            import numpy as np
            self.orientation = np.random.uniform(0, 2 * math.pi)

        # Type-dependent stats
        if worm_type == WormType.WORKER:
            self.max_health = WORM_MAX_HEALTH_WORKER
            self.strength = WORM_STRENGTH_WORKER
            self.vision_radius = WORM_VISION_RADIUS_WORKER
            self.attack_damage = WORM_ATTACK_DAMAGE_WORKER
            self.comm_radius = COMM_RADIUS_WORKER
            self.hidden_dim = WORKER_HIDDEN_DIM
        else:
            self.max_health = WORM_MAX_HEALTH_THINKER
            self.strength = WORM_STRENGTH_THINKER
            self.vision_radius = WORM_VISION_RADIUS_THINKER
            self.attack_damage = WORM_ATTACK_DAMAGE_THINKER
            self.comm_radius = COMM_RADIUS_THINKER
            self.hidden_dim = THINKER_HIDDEN_DIM

        # Pre-compute squared values for fast comparisons
        self.vision_radius_sq = self.vision_radius * self.vision_radius
        self.comm_radius_sq = self.comm_radius * self.comm_radius

        self.health = self.max_health
        self.energy = WORM_MAX_ENERGY
        self.alive = True
        self.infected = False
        self.infection_timer = 0

        # Communication signal vector - no predefined meaning
        self.signal = [0.0] * SIGNAL_DIM  # list for faster element access

        # Attachment graph edges: set of connected worm IDs
        self.attachments: set[int] = set()

        # Local sensor cache (updated each timestep)
        self.nearby_worms: list = []
        self.nearby_enemies: list = []
        self.local_damage_taken: float = 0.0
        self.terrain_type: int = 0

        # Tracking for rewards
        self.damage_blocked_this_step: float = 0.0
        self.flood_kills_this_step: int = 0
        self.was_protected_this_step: bool = False

        # Cached thinker boost (computed once per step)
        self._boost_attack: float = 0.0
        self._boost_move: float = 0.0
        self._boost_comm: float = 0.0

    def distance_to_sq(self, other_x: float, other_y: float) -> float:
        """Squared distance — avoids sqrt for comparison-heavy code."""
        dx = self.x - other_x
        dy = self.y - other_y
        return dx * dx + dy * dy

    def distance_to(self, other_x: float, other_y: float) -> float:
        """Exact distance (only when needed for display/calc)."""
        dx = self.x - other_x
        dy = self.y - other_y
        return math.sqrt(dx * dx + dy * dy)

    def can_attach(self) -> bool:
        return len(self.attachments) < MAX_ATTACHMENTS_PER_WORM and self.energy > 0

    def take_damage(self, amount: float) -> float:
        """Apply damage to this worm. Returns actual damage taken."""
        if not self.alive:
            return 0.0
        actual = min(amount, self.health)
        self.health -= actual
        self.local_damage_taken += actual
        if self.health <= 0:
            self.health = 0
            self.alive = False
        return actual

    def heal(self, amount: float):
        if self.alive:
            self.health = min(self.health + amount, self.max_health)

    def spend_energy(self, amount: float) -> bool:
        """Try to spend energy. Returns True if successful."""
        if self.energy >= amount:
            self.energy -= amount
            return True
        return False

    def regenerate_energy(self, amount: float):
        if self.alive:
            self.energy = min(self.energy + amount, WORM_MAX_ENERGY)

    def clone_state(self) -> dict:
        """Return a serializable snapshot of this worm's state."""
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'type': int(self.worm_type),
            'health': self.health,
            'max_health': self.max_health,
            'energy': self.energy,
            'orientation': self.orientation,
            'signal': list(self.signal),
            'attachments': list(self.attachments),
            'alive': self.alive,
            'infected': self.infected,
        }
