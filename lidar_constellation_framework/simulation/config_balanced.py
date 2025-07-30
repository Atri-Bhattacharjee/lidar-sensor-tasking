# Balanced Configuration for Optimal Speed vs Convergence
# This configuration balances training speed with convergence probability

import os

# =============================================================================
# SIMULATION PARAMETERS (BALANCED)
# =============================================================================
SIMULATION_TIME_STEPS = 35      # Balanced: 30% fewer than current, 75% more than barebones
EPISODE_DURATION_SECONDS = 1200 # Balanced: 20 minutes (33% shorter than current)

# =============================================================================
# CONSTELLATION PARAMETERS (FULL CONSTELLATION)
# =============================================================================
NUM_SATELLITES = 40             # Full constellation (all satellites from .dia files)
SENSOR_SLEW_RATE_DEG_S = 2.0    # Keep same
SENSOR_FOV_DEG = 20.0           # Keep same
SENSOR_MAX_RANGE_M = 1000000    # Keep same

# =============================================================================
# SENSOR NOISE PARAMETERS (OPTIMAL FOR LEARNING)
# =============================================================================
LIDAR_RANGE_SIGMA_M = 0.1       # Keep ultra-low noise
LIDAR_AZIMUTH_SIGMA_DEG = 0.001 # Keep ultra-low noise
LIDAR_ELEVATION_SIGMA_DEG = 0.001 # Keep ultra-low noise
LIDAR_RANGE_RATE_SIGMA_M_S = 0.01 # Keep ultra-low noise
PROBABILITY_OF_DETECTION_MAX = 1.0   # Perfect detection for learning
CLUTTER_RATE_POISSON_LAMBDA = 0.0   # No clutter for clean learning

# =============================================================================
# LMB FILTER PARAMETERS (BALANCED)
# =============================================================================
BIRTH_PROBABILITY = 0.015        # Balanced: 25% lower than current
SURVIVAL_PROBABILITY = 0.999     # Keep high for stability
EXISTENCE_THRESHOLD_PRUNING = 0.001  # Keep same
EXISTENCE_THRESHOLD_EXTRACTION = 0.15 # Balanced: 25% lower than current
PROCESS_NOISE_Q = 0.15          # Balanced: 25% lower than current

# =============================================================================
# GNN & RL PARAMETERS (OPTIMIZED FOR CONVERGENCE)
# =============================================================================
LEARNING_RATE = 5e-4            # Balanced: 67% higher than current, 50% lower than barebones
DISCOUNT_FACTOR_GAMMA = 0.92    # Balanced: 3% lower than current
PPO_CLIP_EPSILON = 0.15         # Balanced: 50% higher than current, 25% lower than barebones
NUM_EPOCHS_PER_UPDATE = 2       # Balanced: 33% lower than current, 100% higher than barebones
BATCH_SIZE = 24                 # Balanced: 25% lower than current, 50% higher than barebones
K_NEAREST_NEIGHBORS = 5         # Balanced: 17% lower than current, 25% higher than barebones
HIDDEN_DIM = 96                 # Balanced: 25% lower than current, 50% higher than barebones
NUM_ATTENTION_HEADS = 2         # Keep same as current (important for attention)

# =============================================================================
# TRAINING PARAMETERS (BALANCED)
# =============================================================================
TOTAL_EPISODES = 100            # Balanced: 50% fewer than current, 100% more than barebones
SAVE_MODEL_EVERY = 15           # Balanced: 40% lower than current
EVALUATION_EVERY = 15           # Balanced: 25% lower than current

# =============================================================================
# FILE PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "ground_truth_data", "ground_truth.cpe")
CONSTELLATION_DATA_PATH = os.path.join(BASE_DIR, "data_generation", "constellation.csv")

# Model and results paths
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "results", "models")
RESULTS_PLOT_PATH = os.path.join(BASE_DIR, "results", "plots")
TRAINING_LOG_PATH = os.path.join(BASE_DIR, "results", "training_logs")

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_PLOT_PATH, exist_ok=True)
os.makedirs(TRAINING_LOG_PATH, exist_ok=True)

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
EARTH_RADIUS_M = 6371000.0
GRAVITATIONAL_CONSTANT = 3.986004418e14
ORBITAL_PERIOD_LEO = 90.0

# =============================================================================
# OSPA PARAMETERS
# =============================================================================
OSPA_CUTOFF_C = 50000.0
OSPA_ORDER_P = 1.0

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"
SEED = 42 