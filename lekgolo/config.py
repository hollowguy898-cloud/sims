"""
Configuration for the Lekgolo Colony Simulation.
Only MAP_SIZE is hard-coded as a constant; everything else is tunable here.
"""

# --- World ---
MAP_WIDTH = 80
MAP_HEIGHT = 80

# --- Simulation ---
MAX_STEPS_PER_EPISODE = 2000
NUM_WORMS_INITIAL = 200
NUM_THINKERS_INITIAL = 20
NUM_FLOOD_INITIAL = 50
FLOOD_SPAWN_RATE = 2          # new Flood per timestep at map edges
MAX_FLOOD_COUNT = 500

# --- Worm Base Properties ---
WORM_VISION_RADIUS_WORKER = 5
WORM_VISION_RADIUS_THINKER = 30
WORM_MAX_HEALTH_WORKER = 100
WORM_MAX_HEALTH_THINKER = 20
WORM_STRENGTH_WORKER = 10
WORM_STRENGTH_THINKER = 1
WORM_MAX_ENERGY = 200
WORM_ENERGY_REGEN = 0.5       # energy recovered per timestep
WORM_MOVE_COST = 1.0
WORM_ATTACK_COST = 3.0
WORM_ATTACH_COST = 0.5
WORM_DETACH_COST = 0.2
WORM_SIGNAL_COST = 0.1
WORM_TRANSFER_COST = 0.1      # fraction lost in transfer
WORM_ATTACK_DAMAGE_WORKER = 15
WORM_ATTACK_DAMAGE_THINKER = 2
WORM_ATTACK_RANGE = 2

# --- Thinker Boosts ---
THINKER_BOOST_ATTACK_ACCURACY = 0.20
THINKER_BOOST_MOVE_EFFICIENCY = 0.20
THINKER_BOOST_COMM_RANGE = 0.20
THINKER_BOOST_RADIUS = 8      # radius within which thinkers boost workers

# --- Attachment ---
MAX_ATTACHMENTS_PER_WORM = 6
ATTACHMENT_MAX_DISTANCE = 2.5
ATTACHMENT_STRENGTH = 1.0     # structural strength per link
DETACH_HEALTH_COST = 5        # damage taken when forcibly detached

# --- Communication ---
SIGNAL_DIM = 4
COMM_RADIUS_WORKER = 8
COMM_RADIUS_THINKER = 30

# --- Reproduction ---
BIOMASS_COST_WORKER = 10
BIOMASS_COST_THINKER = 100
BIOMASS_PER_FLOOD_KILL = 5
BIOMASS_PER_TIMESTEP = 0.5    # passive biomass accumulation
INITIAL_BIOMASS = 200

# --- Flood Agent Properties ---
FLOOD_HEALTH = 30
FLOOD_STRENGTH = 8
FLOOD_SPEED = 1
FLOOD_ATTACK_DAMAGE = 12
FLOOD_INFECTION_DAMAGE = 25
FLOOD_ATTACK_RANGE = 1
FLOOD_VISION = 10
FLOOD_REPRODUCE_INTERVAL = 20  # timesteps between reproduction
FLOOD_REPRODUCE_BIOMASS_COST = 15
FLOOD_SIGNAL_DIM = 4           # Flood communication channels
FLOOD_COMM_RADIUS = 6

# --- Flood RL Agent ---
FLOOD_NUM_DISCRETE_ACTIONS = 5   # move, attack, infect, split, signal
FLOOD_ACTION_PARAM_DIM = 4
FLOOD_HIDDEN_DIM = 32           # smaller than workers - high birth rate, low intelligence
FLOOD_MAX_NEARBY_WORMS = 6
FLOOD_MAX_NEARBY_FLOOD = 6

# --- Rewards (colony-level) ---
REWARD_SURVIVAL_PER_STEP = 0.01
REWARD_FLOOD_KILL = 1.0
REWARD_WORKER_INFECTED = -5.0
REWARD_THINKER_INFECTED = -20.0
REWARD_THINKER_ALIVE_PER_STEP = 0.05
REWARD_CONNECTED_WORKERS_PER_THINKER = 0.01
REWARD_DAMAGE_BLOCKED = 0.1
REWARD_TERRITORY_PER_CELL = 0.001
REWARD_BIOMASS_CHANGE_SCALE = 0.01
REWARD_THINKER_DEATH = -50.0
REWARD_COLONY_FRAGMENTATION = -2.0  # per new fragment when split

# --- Flood Rewards (outcome-based, not behavior-based) ---
FLOOD_REWARD_INFECT_WORKER = 10.0
FLOOD_REWARD_INFECT_THINKER = 30.0
FLOOD_REWARD_COLONY_FRAGMENT = 5.0
FLOOD_REWARD_DEATH = -1.0
FLOOD_REWARD_BIOMASS_GROWTH = 0.5
FLOOD_REWARD_SURVIVAL_PER_STEP = 0.005
FLOOD_REWARD_SPREAD_BONUS = 0.01   # reward for geographic spread

# --- Neural Network ---
WORKER_HIDDEN_DIM = 16
THINKER_HIDDEN_DIM = 128
NUM_DISCRETE_ACTIONS = 6   # move, attach, detach, transfer, attack, signal
ACTION_PARAM_DIM = 5       # continuous parameters per action
OBSERVATION_DIM = None  # computed at runtime based on sensors
ACTION_DIM = None       # computed at runtime

# --- PPO ---
PPO_LEARNING_RATE = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_EPSILON = 0.2
PPO_ENTROPY_COEFF = 0.01
PPO_VALUE_LOSS_COEFF = 0.5
PPO_MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 4
PPO_MINIBATCH_SIZE = 64
PPO_ROLLOUT_LENGTH = 256

# --- Terrain ---
TERRAIN_FLAT = 0
TERRAIN_ROUGH = 1
TERRAIN_WALL = 2
TERRAIN_TOXIC = 3            # deals damage over time
TERRAIN_HIGHGROUND = 4       # vision bonus
TERRAIN_ROUGH_SPEED_PENALTY = 0.5
TERRAIN_WALL_BLOCKS = True
TERRAIN_ROUGH_PROB = 0.15
TERRAIN_WALL_PROB = 0.05
TERRAIN_TOXIC_DAMAGE = 0.5   # damage per step in toxic terrain
TERRAIN_HIGHGROUND_VISION_BONUS = 10  # extra vision radius on high ground

# --- Procedural Generation ---
WORLD_GEN_OCTAVES = 4         # number of noise octaves for heightmap
WORLD_GEN_PERSISTENCE = 0.5   # noise persistence (roughness)
WORLD_GEN_LACUNARITY = 2.0    # noise lacunarity (detail frequency)
WORLD_GEN_SEA_LEVEL = 0.35    # height below which = wall (water/pit)
WORLD_GEN_MOUNTAIN_LEVEL = 0.75  # height above which = high ground
WORLD_GEN_ROUGH_LEVEL = 0.55  # height for rough terrain
WORLD_GEN_MIN_PATHS = 2       # minimum distinct paths between spawn zones
WORLD_GEN_PATH_WIDTH = 3      # minimum corridor width

# --- Map Classifiers ---
MAP_TYPE_OPEN = 'open'
MAP_TYPE_CANYON = 'canyon'
MAP_TYPE_MAZE = 'maze'
MAP_TYPE_ISLAND = 'island'
MAP_TYPE_MIXED = 'mixed'

# --- Dynamic Map Modifiers ---
MODIFIER_TOXIC_ZONE = 'toxic_zone'
MODIFIER_INFECTION_FOG = 'infection_fog'
MODIFIER_COMM_JAM = 'comm_jam'
MODIFIER_BIOMASS_DECAY = 'biomass_decay'
MODIFIER_COLLAPSING = 'collapsing'
NUM_MODIFIERS_PER_MAP = 2         # how many modifiers per generated map
MODIFIER_RADIUS_MIN = 5
MODIFIER_RADIUS_MAX = 15
MODIFIER_INFECTION_FOG_FLOOD_BOOST = 1.5   # Flood damage multiplier in fog
MODIFIER_COMM_JAM_RANGE_PENALTY = 0.3       # comm range multiplier in jam zone
MODIFIER_BIOMASS_DECAY_RATE = 0.1            # biomass lost per step in decay zone

# --- Procedural Events ---
EVENT_FLOOD_SURGE = 'flood_surge'
EVENT_RESOURCE_BLOOM = 'resource_bloom'
EVENT_TERRAIN_COLLAPSE = 'terrain_collapse'
EVENT_THINKER_DISRUPT = 'thinker_disrupt'
EVENT_INTERVAL_MIN = 100       # min timesteps between events
EVENT_INTERVAL_MAX = 300       # max timesteps between events
EVENT_FLOOD_SURGE_COUNT = 20   # number of Flood spawned in surge
EVENT_RESOURCE_BLOOM_AMOUNT = 100
EVENT_TERRAIN_COLLAPSE_RADIUS = 8
EVENT_THINKER_DISRUPT_DURATION = 30  # timesteps
EVENT_THINKER_DISRUPT_BOOST_PENALTY = 0.5  # thinker boost multiplier during disrupt

# --- Biomass Fields ---
BIOMASS_FIELD_COUNT_MIN = 3
BIOMASS_FIELD_COUNT_MAX = 8
BIOMASS_FIELD_RADIUS_MIN = 3
BIOMASS_FIELD_RADIUS_MAX = 8
BIOMASS_FIELD_RATE = 2.0       # biomass gained per step per worm in field

# --- Spawn Formation ---
LEKGLO_SPAWN_RADIUS = 8        # radius of Lekgolo spawn cluster
LEKGLO_THINKER_SPAWN_OFFSET = 2  # thinkers spawn within this distance of center
FLOOD_SPAWN_CLUSTER_COUNT = 3   # number of Flood spawn clusters
FLOOD_SPAWN_CLUSTER_SIZE = 15   # Flood per cluster
FLOOD_SPAWN_ISOLATED_SEEDS = 5  # number of isolated Flood "seeds"
