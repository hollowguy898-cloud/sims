"""
Attachment system - the core mechanic that lets worms form dynamic structures.

Worms can create/break links with neighbors. These links:
- Increase structural strength
- Allow information sharing
- Allow resource sharing
- Restrict movement

The colony becomes a dynamic graph. The RL policy learns whether
staying independent or connecting is better.
"""
import numpy as np
from config import (
    MAX_ATTACHMENTS_PER_WORM, ATTACHMENT_MAX_DISTANCE,
    ATTACHMENT_STRENGTH, DETACH_HEALTH_COST, THINKER_BOOST_RADIUS,
    THINKER_BOOST_ATTACK_ACCURACY, THINKER_BOOST_MOVE_EFFICIENCY,
    THINKER_BOOST_COMM_RANGE
)


class AttachmentSystem:
    """Manages the attachment graph between worms."""

    def __init__(self):
        # edges: dict mapping (id_a, id_b) -> dict with properties
        # Always store with id_a < id_b for consistency
        self.edges: dict[tuple[int, int], dict] = {}

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
        return True

    def remove_edge(self, worm_a_id: int, worm_b_id: int) -> bool:
        """Remove an attachment between two worms."""
        key = self._edge_key(worm_a_id, worm_b_id)
        if key in self.edges:
            del self.edges[key]
            return True
        return False

    def has_edge(self, worm_a_id: int, worm_b_id: int) -> bool:
        key = self._edge_key(worm_a_id, worm_b_id)
        return key in self.edges

    def get_neighbors(self, worm_id: int) -> list[int]:
        """Get all worms attached to the given worm."""
        neighbors = []
        for (a, b), _ in self.edges.items():
            if a == worm_id:
                neighbors.append(b)
            elif b == worm_id:
                neighbors.append(a)
        return neighbors

    def get_edge_strength(self, worm_a_id: int, worm_b_id: int) -> float:
        key = self._edge_key(worm_a_id, worm_b_id)
        if key in self.edges:
            return self.edges[key]['strength']
        return 0.0

    def get_colony_fragments(self, alive_worm_ids: set[int]) -> list[set[int]]:
        """
        Find connected components (fragments) in the attachment graph.
        Returns a list of sets, each set being a connected component.
        """
        if not alive_worm_ids:
            return []

        # Build adjacency for alive worms only
        adjacency: dict[int, set[int]] = {wid: set() for wid in alive_worm_ids}
        for (a, b) in self.edges:
            if a in alive_worm_ids and b in alive_worm_ids:
                adjacency[a].add(b)
                adjacency[b].add(a)

        visited = set()
        fragments = []

        for start_id in alive_worm_ids:
            if start_id in visited:
                continue
            # BFS
            fragment = set()
            queue = [start_id]
            while queue:
                node = queue.pop(0)
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
        More attachments = more structural support = higher damage resistance.
        """
        neighbors = self.get_neighbors(worm_id)
        total_strength = 0.0
        for nid in neighbors:
            total_strength += self.get_edge_strength(worm_id, nid)
        return total_strength

    def compute_damage_reduction(self, worm_id: int) -> float:
        """
        Compute damage reduction factor (0 to 1) based on structural strength.
        Each attachment point gives a small damage reduction.
        """
        structural = self.compute_structural_strength(worm_id)
        # Diminishing returns: each additional attachment gives less
        # reduction = 1 - 1/(1 + structural * 0.1)
        reduction = min(0.5, structural * 0.05)  # cap at 50% reduction
        return reduction

    def constrain_movement(self, worm_id: int, proposed_dx: float,
                           proposed_dy: float, worms_by_id: dict) -> tuple[float, float]:
        """
        Attached worms have restricted movement. The farther an attached neighbor,
        the more the movement is pulled back toward maintaining the link.

        Returns adjusted (dx, dy).
        """
        neighbors = self.get_neighbors(worm_id)
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
        # Spring-like pull: the further from center of attached group,
        # the stronger the pull back
        pull_x = (avg_x - worm.x) * 0.1  # spring constant
        pull_y = (avg_y - worm.y) * 0.1

        # Blend proposed movement with pull
        adjusted_dx = proposed_dx * 0.7 + pull_x
        adjusted_dy = proposed_dy * 0.7 + pull_y

        return adjusted_dx, adjusted_dy

    def tick_edges(self):
        """Age all edges by one timestep."""
        for key in self.edges:
            self.edges[key]['age'] += 1

    def _edge_key(self, id_a: int, id_b: int) -> tuple[int, int]:
        if id_a < id_b:
            return (id_a, id_b)
        return (id_b, id_a)


def compute_thinker_boost(worm, all_worms: list) -> dict:
    """
    Compute the boost a worker receives from nearby thinkers.

    Returns a dict with boost percentages.
    """
    if worm.worm_type == 1:  # Thinkers don't boost themselves
        return {'attack_accuracy': 0.0, 'move_efficiency': 0.0, 'comm_range': 0.0}

    boost = {'attack_accuracy': 0.0, 'move_efficiency': 0.0, 'comm_range': 0.0}
    for other in all_worms:
        if other.id == worm.id or not other.alive:
            continue
        if other.worm_type != 1:  # only thinkers boost
            continue
        d = worm.distance_to(other.x, other.y)
        if d <= THINKER_BOOST_RADIUS:
            # Proximity-based boost (closer = stronger)
            factor = 1.0 - (d / THINKER_BOOST_RADIUS)
            boost['attack_accuracy'] += THINKER_BOOST_ATTACK_ACCURACY * factor
            boost['move_efficiency'] += THINKER_BOOST_MOVE_EFFICIENCY * factor
            boost['comm_range'] += THINKER_BOOST_COMM_RANGE * factor

    # Cap boosts
    boost['attack_accuracy'] = min(boost['attack_accuracy'], 0.6)
    boost['move_efficiency'] = min(boost['move_efficiency'], 0.6)
    boost['comm_range'] = min(boost['comm_range'], 0.6)

    return boost
