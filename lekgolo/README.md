# Lekgolo Colony vs Flood — Emergent Behavior Simulation

Two evolving distributed life systems discovering strategies for surviving each other.

## What Is This?

A multi-agent reinforcement learning simulation where two asymmetric lifeforms co-evolve:

| | Lekgolo Colony | Flood |
|---|---|---|
| **Strategy** | Structure + Coordination | Replication + Disruption |
| **Agents** | Workers (strong, cheap) + Thinkers (fragile, expensive, smart) | Replicating parasites |
| **Advantage** | Long-range coordination, spatial intelligence, structural defense | High birth rate, infection pressure, expendable units |
| **Reward scaling** | Structure preservation, thinker survival | Geographic spread, colony fragmentation |

Neither side is scripted. Neither has predefined strategies. Behaviors **emerge** from the reward structure and physical constraints.

## Emergent Behaviors (Not Programmed)

The simulation does NOT contain any of these behaviors as code:

- `become_wall()` / `become_tower()` / `become_hunter()` — these do not exist
- `guard_thinker()` — no rule says workers should protect thinkers
- `form_brain()` — no rule says thinkers should cluster
- `swarm_weak_point()` — no rule says Flood should probe defenses

Instead, the reward function targets **outcomes**, and the RL agents discover strategies:

**Expected Lekgolo emergence:**
- Defensive walls around thinkers
- Kill-boxes at choke points
- Mobile "muscle" structures that push enemies
- Shield layers with sacrificial outer shells
- Brain cores: thinkers buried inside worker fortresses
- Infected segment amputation (cut off infected parts of the colony)

**Expected Flood emergence:**
- Probing attacks testing for weak points
- Swarming to isolate thinkers from workers
- Sacrificial infection units
- Infection pressure fronts (dense waves pushing)
- Targeting communication links between colony segments

## Architecture

### Worm Agent (Lekgolo)

Each worm is an individual RL agent with:

**State:** Position, orientation, energy, health, infection status, signal vector

**Sensors (local only):**
- Nearby worms (positions, types, health, signals)
- Nearby enemies (positions, health)
- Terrain type
- Structural strength from attachments
- Thinker boost levels

**Actions:**
| Action | Description |
|--------|-------------|
| Move | Direction + speed |
| Attach | Link to nearby worm |
| Detach | Break link (costs health) |
| Transfer | Send energy to neighbor |
| Attack | Damage nearby Flood |
| Signal | Set 4D communication vector |

**No action called:** `become_barrier`, `become_hunter`, `become_tower` — those emerge.

### Caste System

| | Worker | Thinker |
|---|---|---|
| Health | 100 | 20 |
| Strength | 10 | 1 |
| Vision | 5 cells | 30 cells |
| Attack | 15 damage | 2 damage |
| Neural net | 16 hidden | 128 hidden |
| Biomass cost | 10 | 100 |

Same architecture, different capacity. Thinkers don't have predefined "commander" behavior — they simply have more ability to process information. The colony discovers what thinkers are useful for.

### Attachment Physics

The core mechanic. Each worm can create links:

```
Worm A <-> Worm B
```

Links:
- Increase structural strength (damage reduction)
- Allow information sharing (communication through structure)
- Allow resource sharing (energy transfer)
- Restrict movement (spring-like tethering)

Colony = dynamic graph. RL learns: is it better to stay independent or connect?

**Emergent structure properties:**

| Shape | Emergent Advantage |
|-------|--------------------|
| Dense cluster | High durability |
| Long wall | Blocks Flood |
| Ring | Protects center |
| Tall stack | Extended vision |
| Distributed swarm | Hard to wipe out |

### Communication

Each worm has a 4D signal vector: `signal = [0.0, 0.0, 0.0, 0.0]`

Neighbors can read it. **Don't define meaning.** The network learns meaning.

You may eventually observe:
- `signal[0]` = danger
- `signal[1]` = gather
- `signal[2]` = attack
- `signal[3]` = retreat

Except nobody told them that.

### Flood Agent

Flood are RL agents, not scripted enemies. Each Flood unit has:

**State:** Position, health, biomass, signal vector

**Observations:**
- Nearby worms (with vulnerability signals: isolated thinkers, thin walls)
- Nearby Flood (for coordination)
- Local biomass availability
- Colony density
- Vulnerability signals

**Actions:**
| Action | Description |
|--------|-------------|
| Move | Direction + speed |
| Attack | Damage nearest worm |
| Infect | 1:1 conversion attempt (30% chance) |
| Split | Spawn new Flood (costs biomass) |
| Signal | Set communication vector |

Flood are deliberately simpler: 32 hidden units vs 16/128 for worms. They compensate with numbers and replication.

### Reward Design (The Most Important Part)

**Lekgolo rewards — outcomes only:**

| Reward | Value | Targets |
|--------|-------|---------|
| Survival | +0.01/worm/step | Don't die |
| Flood kills | +1.0/kill | Fight back |
| Thinker alive | +0.05/thinker/step | Thinkers are valuable |
| Connectivity | +0.01/connected worker | Stay attached |
| Damage blocked | +0.1/blocked | Structures help |
| Worker infected | -5.0 | Infection is bad |
| Thinker infected | -20.0 | Thinker infection is worse |
| Thinker death | -50.0 | Losing thinkers is catastrophic |
| Fragmentation | -2.0/fragment | Don't split |

**Flood rewards — outcomes only:**

| Reward | Value | Targets |
|--------|-------|---------|
| Survival | +0.005/flood/step | Exist (cheap) |
| Infect worker | +10.0 | Penetration |
| Infect thinker | +30.0 | Disruption |
| Colony fragment | +5.0 | Break their structure |
| Biomass growth | +0.5 | Replicate |
| Geographic spread | +0.01/area | Control territory |
| Death | -1.0 | (small penalty — expendable) |

**Notice:** Never rewarded: "protect thinkers", "build walls", "form brains", "swarm thinkers", "probe defenses". Those are behaviors. Rewards target outcomes. If rewards are correct, the colony invents strategies neither of us can predict.

## Procedural Generation

Every match begins from a seed. Same seed = replayable fight. Different seed = new world.

### Pipeline

```
seed
 ↓
heightmap (multi-octave noise)
 ↓
terrain (threshold → flat/rough/wall/highground/toxic)
 ↓
region classification (connected components)
 ↓
path validation (ensure 2-4 paths between spawn zones)
 ↓
spawn formation placement
 ↓
biomass field generation (scarcity gradients)
 ↓
map modifier injection (1-2 per map)
 ↓
event schedule generation
 ↓
map classification tag (open/canyon/maze/island/mixed)
```

### Terrain Types

| Type | Effect |
|------|--------|
| Flat | Normal movement |
| Rough | Slowed movement |
| Wall | Impassable |
| Toxic | Damage over time |
| High Ground | +10 vision radius |

### Spawn Formations

**Lekgolo:** Clustered near center — thinkers at core, workers on perimeter.

**Flood:** Distributed clusters near edges + isolated "infection seeds" with density gradients.

This alone creates natural conflict dynamics.

### Biomass Fields

Scarcity gradients force movement:
- **Safe fields** (near Lekgolo spawn): low yield
- **Contested fields** (mid-map): high yield but risky

Without this, agents just camp.

### Dynamic Map Modifiers (1-2 per map)

| Modifier | Effect |
|----------|--------|
| Toxic Zone | Damages all entities in area |
| Infection Fog | Boosts Flood damage 1.5x in area |
| Comm Jam | Reduces communication range to 30% in area |
| Biomass Decay | Drains biomass in area |
| Collapsing | Terrain gradually turns to wall |

### Procedural Events (during match)

| Event | Effect |
|-------|--------|
| Flood Surge | 20 Flood spawn at edges |
| Resource Bloom | +100 biomass |
| Terrain Collapse | Random area becomes wall |
| Thinker Disrupt | Thinker boosts halved for 30 steps |

### Map Classification

Each map is tagged: `open`, `canyon`, `maze`, `island`, or `mixed`

This enables analysis: "Flood wins in open maps, Lekgolo wins in canyon maps" — balance tuning becomes scientific.

## Training

PPO (Proximal Policy Optimization) with GAE for both sides.

**Lekgolo:** Shared policy (workers share one 16-hidden network, thinkers share one 128-hidden network)

**Flood:** Shared policy (32-hidden network, simpler — they compensate with numbers)

```bash
# Install dependencies
pip install torch numpy matplotlib

# Train (from the lekgolo/ directory)
cd lekgolo
python train.py --episodes 1000

# Render a snapshot
python visualize.py --mode snapshot

# Analyze training
python visualize.py --mode analyze
```

## File Structure

```
lekgolo/
├── config.py              # All tunable parameters (only MAP_SIZE hard-coded)
├── worm.py                # Lekgolo worm agent (Worker/Thinker)
├── flood.py               # Flood RL agent + observations + action decoding
├── world_gen.py           # Full procedural generation pipeline
├── terrain.py             # Backward-compat wrapper around world_gen
├── attachment_system.py   # Dynamic graph of worm connections
├── sensors.py             # Observation vectors for worms
├── actions.py             # 6 worm actions (move/attach/detach/transfer/attack/signal)
├── rewards.py             # Outcome-based rewards for both sides
├── network.py             # Policy networks (Worker/Thinker/Flood)
├── ppo_trainer.py         # PPO with GAE, clipped surrogate, minibatch updates
├── environment.py         # Main simulation loop
├── train.py               # Training script
└── visualize.py           # Rendering and analysis
```

## Key Design Principles

1. **The worm is the agent.** Not "hunter units" or "barrier units."
2. **No hard-coded forms.** `become_wall()` does not exist. Structures emerge.
3. **Rewards target outcomes, not behaviors.** Never "form wall." Always "damage blocked."
4. **Thinkers are vulnerable.** 20 HP vs 100 HP. Protection must emerge.
5. **Communication has no predefined meaning.** 4D signal vectors. The network learns meaning.
6. **Flood are agents, not scripts.** They learn too. Arms race is real.
7. **Asymmetry is enforced.** Lekgolo = structure + coordination. Flood = replication + disruption.
8. **Procedural generation forces adaptation.** No overfitting to static strategies.

## The Weird Stuff

RL sometimes finds solutions that look bizarre but work extremely well. The colony might form giant rotating "blenders" of worms because the reward function accidentally made that optimal. Evolution has no dignity. It only cares whether the reward number goes up.

If after 50 million simulation steps the worms invent a moving armored brain-train that rolls across the map while ejecting infected segments — that's a discovery. If you explicitly reward "make a wall," you've already decided the answer before the experiment begins.

## Dependencies

- Python 3.12+
- PyTorch (CPU)
- NumPy
- SciPy
- Matplotlib
- Pygame (optional, for interactive rendering)

## License

See LICENSE file.
