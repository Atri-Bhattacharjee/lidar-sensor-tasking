# Barebones Configuration for Fastest Training
# This configuration minimizes everything for speed while maintaining functionality

import os

# =============================================================================
# SIMULATION PARAMETERS (MINIMAL)
# =============================================================================
SIMULATION_TIME_STEPS = 20      # Reduced from 50 (60% fewer timesteps)
EPISODE_DURATION_SECONDS = 900  # Reduced from 1800 (15 minutes)

# =============================================================================
# CONSTELLATION PARAMETERS (FULL CONSTELLATION)
# =============================================================================
NUM_SATELLITES = 40             # Full constellation (all satellites from .dia files)
SENSOR_SLEW_RATE_DEG_S = 2.0    # Keep same
SENSOR_FOV_DEG = 10.0           # Keep same
SENSOR_MAX_RANGE_M = 1000000    # Keep same

# =============================================================================
# SENSOR NOISE PARAMETERS (MINIMAL)
# =============================================================================
LIDAR_RANGE_SIGMA_M = 0.1       # Keep ultra-low noise
LIDAR_AZIMUTH_SIGMA_DEG = 0.001 # Keep ultra-low noise
LIDAR_ELEVATION_SIGMA_DEG = 0.001 # Keep ultra-low noise
LIDAR_RANGE_RATE_SIGMA_M_S = 0.01 # Keep ultra-low noise
PROBABILITY_OF_DETECTION_MAX = 1.0   # Perfect detection
CLUTTER_RATE_POISSON_LAMBDA = 0.0   # No clutter

# =============================================================================
# LMB FILTER PARAMETERS (MINIMAL)
# =============================================================================
BIRTH_PROBABILITY = 0.01         # Reduced from 0.02
SURVIVAL_PROBABILITY = 0.999     # Increased from 0.998
EXISTENCE_THRESHOLD_PRUNING = 0.001  # Keep same
EXISTENCE_THRESHOLD_EXTRACTION = 0.1  # Reduced from 0.2
PROCESS_NOISE_Q = 0.1           # Reduced from 0.2

# =============================================================================
# GNN & RL PARAMETERS (MINIMAL)
# =============================================================================
LEARNING_RATE = 1e-3            # Increased from 3e-4 (faster learning)
DISCOUNT_FACTOR_GAMMA = 0.9     # Reduced from 0.95
PPO_CLIP_EPSILON = 0.2          # Increased from 0.1
NUM_EPOCHS_PER_UPDATE = 1       # Reduced from 3
BATCH_SIZE = 16                 # Reduced from 32
K_NEAREST_NEIGHBORS = 4         # Reduced from 6
HIDDEN_DIM = 64                 # Reduced from 128
NUM_ATTENTION_HEADS = 1         # Reduced from 2

# =============================================================================
# TRAINING PARAMETERS (MINIMAL)
# =============================================================================
TOTAL_EPISODES = 50             # Reduced from 200 (75% fewer episodes)
SAVE_MODEL_EVERY = 10           # Reduced from 25
EVALUATION_EVERY = 10           # Reduced from 20

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