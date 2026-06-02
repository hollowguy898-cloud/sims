"""
Lekgolo Worm agent.
Each worm is an individual agent with its own state, sensors, and policy.
Worm types (Worker / Thinker) differ in capacity, not in architecture.
"""
import numpy as np
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

    def __init__(self, x: float, y: float, worm_type: WormType = WormType.WORKER):
        self.id = Worm._next_id
        Worm._next_id += 1

        self.x = x
        self.y = y
        self.worm_type = worm_type

        # Orientation as angle in radians
        self.orientation = np.random.uniform(0, 2 * np.pi)

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

        self.health = self.max_health
        self.energy = WORM_MAX_ENERGY
        self.alive = True
        self.infected = False
        self.infection_timer = 0

        # Communication signal vector - no predefined meaning
        self.signal = np.zeros(SIGNAL_DIM, dtype=np.float32)

        # Attachment graph edges: set of connected worm IDs
        self.attachments: set[int] = set()

        # Local sensor cache (updated each timestep)
        self.nearby_worms: list['Worm'] = []
        self.nearby_enemies: list = []  # list of FloodOrganism
        self.local_damage_taken: float = 0.0
        self.terrain_type: int = 0

        # Tracking for rewards
        self.damage_blocked_this_step: float = 0.0
        self.flood_kills_this_step: int = 0
        self.was_protected_this_step: bool = False

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    @position.setter
    def position(self, val: np.ndarray):
        self.x = float(val[0])
        self.y = float(val[1])

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

    def distance_to(self, other_x: float, other_y: float) -> float:
        return np.sqrt((self.x - other_x) ** 2 + (self.y - other_y) ** 2)

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
            'signal': self.signal.copy(),
            'attachments': list(self.attachments),
            'alive': self.alive,
            'infected': self.infected,
        }
