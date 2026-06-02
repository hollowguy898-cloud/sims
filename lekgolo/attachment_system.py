"""
Attachment system - the core mechanic that lets worms form dynamic structures.

Worms can create/break links with neighbors. These links:
- Increase structural strength
- Allow information sharing
- Allow resource sharing
- Restrict movement

The colony becomes a dynamic graph. The RL policy learns whether
staying independent or connecting is better.

OPTIMIZED: Uses adjacency index for O(1) neighbor lookups,
deque for BFS, and batch thinker boost computation.
"""
import math
from collections import deque
from config import (
    MAX_ATTACHMENTS_PER_WORM, ATTACHMENT_MAX_DISTANCE,
    ATTACHMENT_STRENGTH, DETACH_HEALTH_COST, THINKER_BOOST_RADIUS,
    THINKER_BOOST_ATTACK_ACCURACY, THINKER_BOOST_MOVE_EFFICIENCY,
    THINKER_BOOST_COMM_RANGE
)

# Pre-compute squared boost radius for fast comparisons
_THINKER_BOOST_RADIUS_SQ = THINKER_BOOST_RADIUS * THINKER_BOOST_RADIUS


class AttachmentSystem:
    """Manages the attachment graph between worms."""

    def __init__(self):
        # edges: dict mapping (id_a, id_b) -> dict with properties
        # Always store with id_a < id_b for consistency
        self.edges: dict[tuple[int, int], dict] = {}
        # Adjacency index: worm_id -> set of neighbor ids
        # Maintained in sync with edges for O(1) lookups
        self.adjacency: dict[int, set[int]] = {}

    def add_edge(self, worm_a_id: int, worm_b_id: int,
                 strength: float = ATTACHMENT_STRENGTH) -> bool:
        """Create an attachment between two worms."""
        key = self._edge_key(worm_a_id, worm_b_id)
        if key in self.edges:
            return True  # already attached
        self.edges[key] = {
            'strength': strength,
            'age': 0,
        }
        # Update adjacency index
        if worm_a_id not in self.adjacency:
            self.adjacency[worm_a_id] = set()
        if worm_b_id not in self.adjacency:
            self.adjacency[worm_b_id] = set()
        self.adjacency[worm_a_id].add(worm_b_id)
        self.adjacency[worm_b_id].add(worm_a_id)
        return True

    def remove_edge(self, worm_a_id: int, worm_b_id: int) -> bool:
        """Remove an attachment between two worms."""
        key = self._edge_key(worm_a_id, worm_b_id)
        if key in self.edges:
            del self.edges[key]
            # Update adjacency index
            if worm_a_id in self.adjacency:
                self.adjacency[worm_a_id].discard(worm_b_id)
                if not self.adjacency[worm_a_id]:
                    del self.adjacency[worm_a_id]
            if worm_b_id in self.adjacency:
                self.adjacency[worm_b_id].discard(worm_a_id)
                if not self.adjacency[worm_b_id]:
                    del self.adjacency[worm_b_id]
            return True
        return False

    def has_edge(self, worm_a_id: int, worm_b_id: int) -> bool:
        key = self._edge_key(worm_a_id, worm_b_id)
        return key in self.edges

    def get_neighbors(self, worm_id: int) -> list[int]:
        """Get all worms attached to the given worm. O(1) via adjacency index."""
        return list(self.adjacency.get(worm_id, set()))

    def get_edge_strength(self, worm_a_id: int, worm_b_id: int) -> float:
        key = self._edge_key(worm_a_id, worm_b_id)
        if key in self.edges:
            return self.edges[key]['strength']
        return 0.0

    def get_colony_fragments(self, alive_worm_ids: set[int]) -> list[set[int]]:
        """
        Find connected components (fragments) in the attachment graph.
        Returns a list of sets, each set being a connected component.
        Uses deque for O(1) BFS operations.
        """
        if not alive_worm_ids:
            return []

        # Build adjacency for alive worms only (from index)
        adjacency: dict[int, set[int]] = {}
        for (a, b) in self.edges:
            if a in alive_worm_ids and b in alive_worm_ids:
                if a not in adjacency:
                    adjacency[a] = set()
                if b not in adjacency:
                    adjacency[b] = set()
                adjacency[a].add(b)
                adjacency[b].add(a)

        visited = set()
        fragments = []

        for start_id in alive_worm_ids:
            if start_id in visited:
                continue
            # BFS with deque for O(1) popleft
            fragment = set()
            queue = deque([start_id])
            while queue:
                node = queue.popleft()  # O(1) instead of O(n)
                if node in visited:
                    continue
                visited.add(node)
                fragment.add(node)
                for neighbor in adjacency.get(node, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            fragments.append(fragment)

        return fragments

    def count_fragments(self, alive_worm_ids: set[int]) -> int:
        """Count the number of disconnected colony fragments."""
        return len(self.get_colony_fragments(alive_worm_ids))

    def compute_structural_strength(self, worm_id: int) -> float:
        """
        Compute structural strength bonus for a worm based on its attachments.
        O(degree) via adjacency index instead of O(E).
        """
        neighbors = self.adjacency.get(worm_id, set())
        total_strength = 0.0
        for nid in neighbors:
            total_strength += self.get_edge_strength(worm_id, nid)
        return total_strength

    def compute_damage_reduction(self, worm_id: int) -> float:
        """
        Compute damage reduction factor (0 to 1) based on structural strength.
        """
        structural = self.compute_structural_strength(worm_id)
        reduction = min(0.5, structural * 0.05)  # cap at 50% reduction
        return reduction

    def constrain_movement(self, worm_id: int, proposed_dx: float,
                           proposed_dy: float, worms_by_id: dict) -> tuple[float, float]:
        """
        Attached worms have restricted movement.
        O(degree) via adjacency index instead of O(E).
        """
        neighbors = self.adjacency.get(worm_id, set())
        if not neighbors:
            return proposed_dx, proposed_dy

        # Pull toward average position of attached neighbors
        avg_x, avg_y = 0.0, 0.0
        count = 0
        for nid in neighbors:
            if nid in worms_by_id and worms_by_id[nid].alive:
                avg_x += worms_by_id[nid].x
                avg_y += worms_by_id[nid].y
                count += 1

        if count == 0:
            return proposed_dx, proposed_dy

        avg_x /= count
        avg_y /= count

        worm = worms_by_id[worm_id]
        pull_x = (avg_x - worm.x) * 0.1
        pull_y = (avg_y - worm.y) * 0.1

        adjusted_dx = proposed_dx * 0.7 + pull_x
        adjusted_dy = proposed_dy * 0.7 + pull_y

        return adjusted_dx, adjusted_dy

    def tick_edges(self):
        """Age all edges by one timestep."""
        for props in self.edges.values():
            props['age'] += 1

    def _edge_key(self, id_a: int, id_b: int) -> tuple[int, int]:
        if id_a < id_b:
            return (id_a, id_b)
        return (id_b, id_a)


def compute_thinker_boost(worm, all_worms: list) -> dict:
    """
    Compute the boost a worker receives from nearby thinkers.
    Uses squared distance to avoid sqrt.
    """
    if worm.worm_type == 1:
        return {'attack_accuracy': 0.0, 'move_efficiency': 0.0, 'comm_range': 0.0}

    boost_attack = 0.0
    boost_move = 0.0
    boost_comm = 0.0
    wx, wy = worm.x, worm.y

    for other in all_worms:
        if other.id == worm.id or not other.alive:
            continue
        if other.worm_type != 1:
            continue
        dx = other.x - wx
        dy = other.y - wy
        d_sq = dx * dx + dy * dy
        if d_sq <= _THINKER_BOOST_RADIUS_SQ:
            d = math.sqrt(d_sq)
            factor = 1.0 - (d / THINKER_BOOST_RADIUS)
            boost_attack += THINKER_BOOST_ATTACK_ACCURACY * factor
            boost_move += THINKER_BOOST_MOVE_EFFICIENCY * factor
            boost_comm += THINKER_BOOST_COMM_RANGE * factor

    return {
        'attack_accuracy': min(boost_attack, 0.6),
        'move_efficiency': min(boost_move, 0.6),
        'comm_range': min(boost_comm, 0.6),
    }


def compute_all_thinker_boosts(alive_worms: list) -> dict[int, dict]:
    """
    Compute thinker boosts for ALL worms in one pass.
    O(W * T) instead of O(W^2) — only scans thinkers for each worm.
    Returns {worm_id: boost_dict}.
    """
    thinkers = [w for w in alive_worms if w.alive and w.worm_type == 1]
    if not thinkers:
        return {w.id: {'attack_accuracy': 0.0, 'move_efficiency': 0.0, 'comm_range': 0.0}
                for w in alive_worms}

    # Pre-extract thinker positions
    thinker_positions = [(t.x, t.y) for t in thinkers]

    boosts = {}
    for worm in alive_worms:
        if worm.worm_type == 1:
            boosts[worm.id] = {'attack_accuracy': 0.0, 'move_efficiency': 0.0, 'comm_range': 0.0}
            continue

        boost_attack = 0.0
        boost_move = 0.0
        boost_comm = 0.0
        wx, wy = worm.x, worm.y

        for tx, ty in thinker_positions:
            dx = tx - wx
            dy = ty - wy
            d_sq = dx * dx + dy * dy
            if d_sq <= _THINKER_BOOST_RADIUS_SQ:
                d = math.sqrt(d_sq)
                factor = 1.0 - (d / THINKER_BOOST_RADIUS)
                boost_attack += THINKER_BOOST_ATTACK_ACCURACY * factor
                boost_move += THINKER_BOOST_MOVE_EFFICIENCY * factor
                boost_comm += THINKER_BOOST_COMM_RANGE * factor

        boosts[worm.id] = {
            'attack_accuracy': min(boost_attack, 0.6),
            'move_efficiency': min(boost_move, 0.6),
            'comm_range': min(boost_comm, 0.6),
        }
    return boosts
