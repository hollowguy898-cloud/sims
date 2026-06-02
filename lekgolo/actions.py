"""
Action system for Lekgolo worms.

Actions:
  0: Move (dx, dy continuous)
  1: Attach to nearby worm
  2: Detach from attached worm
  3: Transfer energy to attached/nearby worm
  4: Attack nearby enemy
  5: Signal (set communication vector)

OPTIMIZED: Removed unnecessary .copy(), uses squared distance
for range checks, and cached thinker boosts.
"""
import math
import numpy as np
from config import (
    WORM_MOVE_COST, WORM_ATTACH_COST, WORM_DETACH_COST,
    WORM_ATTACK_COST, WORM_SIGNAL_COST, WORM_TRANSFER_COST,
    WORM_ATTACK_RANGE, ATTACHMENT_MAX_DISTANCE, MAX_ATTACHMENTS_PER_WORM,
    SIGNAL_DIM, DETACH_HEALTH_COST,
    NUM_DISCRETE_ACTIONS, ACTION_PARAM_DIM
)

ACTION_VECTOR_DIM = 1 + ACTION_PARAM_DIM

# Pre-compute squared distances for fast comparisons
_ATTACK_RANGE_SQ = WORM_ATTACK_RANGE * WORM_ATTACK_RANGE
_ATTACH_MAX_DIST_SQ = ATTACHMENT_MAX_DISTANCE * ATTACHMENT_MAX_DISTANCE
_ATTACH_TETHER_SQ = (ATTACHMENT_MAX_DISTANCE * 1.5) ** 2


def decode_action(action_vector: np.ndarray):
    """
    Decode a continuous action vector into a structured action dict.
    No .copy() — the params slice is consumed immediately.
    """
    action_type = int(np.clip(np.round(action_vector[0]), 0, NUM_DISCRETE_ACTIONS - 1))
    params = action_vector[1:]

    return {
        'type': action_type,
        'params': params,  # no .copy() — consumed immediately
    }


def execute_move(worm, dx: float, dy: float, terrain: np.ndarray,
                 attachment_system, worms_by_id: dict,
                 thinker_boost: dict) -> bool:
    """Execute a move action for a worm."""
    if not worm.alive or not worm.spend_energy(WORM_MOVE_COST):
        return False

    efficiency = 1.0 + thinker_boost.get('move_efficiency', 0.0)
    actual_cost = WORM_MOVE_COST / efficiency

    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-6:
        return False

    step_size = min(1.0, dist)
    norm_dx = (dx / dist) * step_size
    norm_dy = (dy / dist) * step_size

    norm_dx, norm_dy = attachment_system.constrain_movement(
        worm.id, norm_dx, norm_dy, worms_by_id
    )

    new_x = worm.x + norm_dx
    new_y = worm.y + norm_dy

    from terrain import is_passable, movement_cost
    grid_x = int(np.clip(np.round(new_x), 0, terrain.shape[1] - 1))
    grid_y = int(np.clip(np.round(new_y), 0, terrain.shape[0] - 1))

    if not is_passable(terrain, grid_x, grid_y):
        return False

    move_mult = movement_cost(terrain, grid_x, grid_y)
    if move_mult == float('inf'):
        return False

    actual_cost *= move_mult
    if worm.energy + actual_cost - WORM_MOVE_COST > 0:
        worm.spend_energy(max(0, actual_cost - WORM_MOVE_COST))

    # Check attachment tether — using squared distance
    for nid in worm.attachments:
        if nid in worms_by_id and worms_by_id[nid].alive:
            partner = worms_by_id[nid]
            pdx = new_x - partner.x
            pdy = new_y - partner.y
            new_dist_sq = pdx * pdx + pdy * pdy
            if new_dist_sq > _ATTACH_TETHER_SQ:
                direction_x = new_x - worm.x
                direction_y = new_y - worm.y
                current_dist = worm.distance_to(partner.x, partner.y)
                max_allowed = ATTACHMENT_MAX_DISTANCE * 1.5 - current_dist
                if max_allowed > 0:
                    scale = max_allowed / max(dist, 1e-6)
                    new_x = worm.x + direction_x * scale
                    new_y = worm.y + direction_y * scale
                else:
                    return False

    new_x = np.clip(new_x, 0, terrain.shape[1] - 1)
    new_y = np.clip(new_y, 0, terrain.shape[0] - 1)

    worm.x = new_x
    worm.y = new_y

    if abs(norm_dx) > 1e-6 or abs(norm_dy) > 1e-6:
        worm.orientation = math.atan2(norm_dy, norm_dx)

    return True


def execute_attach(worm, target_idx: int, all_worms: list,
                   attachment_system) -> bool:
    """Attempt to attach to a nearby worm."""
    if not worm.alive or not worm.can_attach():
        return False
    if not worm.spend_energy(WORM_ATTACH_COST):
        return False

    if not worm.nearby_worms or target_idx < 0:
        return False

    target_idx = int(np.clip(target_idx, 0, len(worm.nearby_worms) - 1))
    target = worm.nearby_worms[target_idx]

    if not target.alive or not target.can_attach():
        return False

    d_sq = worm.distance_to_sq(target.x, target.y)
    if d_sq > _ATTACH_MAX_DIST_SQ:
        return False

    attachment_system.add_edge(worm.id, target.id)
    worm.attachments.add(target.id)
    target.attachments.add(worm.id)

    return True


def execute_detach(worm, target_idx: int, attachment_system,
                   worms_by_id: dict) -> bool:
    """Detach from an attached worm."""
    if not worm.alive:
        return False
    if not worm.spend_energy(WORM_DETACH_COST):
        return False

    attached_list = list(worm.attachments)
    if not attached_list or target_idx < 0:
        return False

    target_idx = int(np.clip(target_idx, 0, len(attached_list) - 1))
    target_id = attached_list[target_idx]

    attachment_system.remove_edge(worm.id, target_id)
    worm.attachments.discard(target_id)

    if target_id in worms_by_id:
        worms_by_id[target_id].attachments.discard(worm.id)

    worm.take_damage(DETACH_HEALTH_COST)

    return True


def execute_transfer(worm, target_idx: int, amount: float,
                     all_worms: list, worms_by_id: dict) -> bool:
    """Transfer energy to a nearby or attached worm."""
    if not worm.alive:
        return False

    amount = float(np.clip(amount, 0, 50))

    if not worm.nearby_worms and not worm.attachments:
        return False

    candidates = []
    for wid in worm.attachments:
        if wid in worms_by_id and worms_by_id[wid].alive:
            candidates.append(worms_by_id[wid])
    for w in worm.nearby_worms:
        if w.alive and w.id not in worm.attachments:
            candidates.append(w)

    if not candidates:
        return False

    target_idx = int(np.clip(target_idx, 0, len(candidates) - 1))
    target = candidates[target_idx]

    d_sq = worm.distance_to_sq(target.x, target.y)
    max_range = max(ATTACHMENT_MAX_DISTANCE, worm.comm_radius)
    if d_sq > max_range * max_range:
        return False

    actual_transfer = amount * (1.0 - WORM_TRANSFER_COST)
    if worm.energy < amount:
        actual_transfer = worm.energy * (1.0 - WORM_TRANSFER_COST)
        amount = worm.energy

    if amount <= 0:
        return False

    worm.spend_energy(amount)
    target.energy = min(target.energy + actual_transfer, 200.0)

    return True


def execute_attack(worm, target_idx: int, flood_list: list,
                   thinker_boost: dict, rng=None) -> bool:
    """Attack a nearby Flood organism. Uses squared distance for range check."""
    if not worm.alive:
        return False
    if not worm.spend_energy(WORM_ATTACK_COST):
        return False

    if not worm.nearby_enemies or target_idx < 0:
        return False

    target_idx = int(np.clip(target_idx, 0, len(worm.nearby_enemies) - 1))
    target = worm.nearby_enemies[target_idx]

    if not target.alive:
        return False

    d_sq = worm.distance_to_sq(target.x, target.y)
    if d_sq > _ATTACK_RANGE_SQ:
        return False

    base_damage = worm.attack_damage
    accuracy = 0.7 + thinker_boost.get('attack_accuracy', 0.0)

    if rng is not None:
        hit = rng.random() < accuracy
    else:
        hit = np.random.random() < accuracy

    if hit:
        damage = base_damage
    else:
        damage = base_damage * 0.3

    actual_damage = target.take_damage(damage)

    if not target.alive:
        worm.flood_kills_this_step += 1

    return True


def execute_signal(worm, signal_values: np.ndarray) -> bool:
    """Set this worm's communication signal vector."""
    if not worm.alive:
        return False
    if not worm.spend_energy(WORM_SIGNAL_COST):
        return False

    for i in range(min(len(signal_values), SIGNAL_DIM)):
        worm.signal[i] = max(-1.0, min(1.0, float(signal_values[i])))

    return True


def process_action(worm, action_dict: dict, all_worms: list,
                   flood_list: list, terrain: np.ndarray,
                   attachment_system, worms_by_id: dict,
                   thinker_boost: dict, rng=None) -> bool:
    """Process a decoded action for a worm."""
    action_type = action_dict['type']
    params = action_dict['params']

    if action_type == 0:
        return execute_move(worm, params[0], params[1], terrain,
                            attachment_system, worms_by_id, thinker_boost)
    elif action_type == 1:
        return execute_attach(worm, params[0], all_worms, attachment_system)
    elif action_type == 2:
        return execute_detach(worm, params[0], attachment_system, worms_by_id)
    elif action_type == 3:
        return execute_transfer(worm, params[0], params[1] * 50,
                                all_worms, worms_by_id)
    elif action_type == 4:
        return execute_attack(worm, params[0], flood_list, thinker_boost, rng)
    elif action_type == 5:
        return execute_signal(worm, params[1:1 + SIGNAL_DIM])
    else:
        return False
