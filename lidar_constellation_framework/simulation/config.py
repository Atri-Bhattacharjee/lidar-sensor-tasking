# Ultra-Fast Configuration for 1-2 Hour Training
# This configuration aggressively optimizes for speed while maintaining learning capability

import os

# =============================================================================
# SIMULATION PARAMETERS (BALANCED FOR LEARNING)
# =============================================================================
SIMULATION_TIME_STEPS = 50      # Increased from 25 (100% longer episodes for better learning)
EPISODE_DURATION_SECONDS = 1800  # Increased from 900 (30 minutes for more context)

# =============================================================================
# CONSTELLATION PARAMETERS (FULL CONSTELLATION)
# =============================================================================
NUM_SATELLITES = 40             # Full constellation (all satellites)
SENSOR_SLEW_RATE_DEG_S = 2.0    # Keep same
SENSOR_FOV_DEG = 60.0           # Increased from 20.0 (much wider FOV for better detection)
SENSOR_MAX_RANGE_M = 1000000    # Keep same

# =============================================================================
# SENSOR NOISE PARAMETERS (ULTRA-LOW NOISE FOR DIAGNOSTIC)
# =============================================================================
LIDAR_RANGE_SIGMA_M = 0.1       # Ultra-low range noise (95% reduction)
LIDAR_AZIMUTH_SIGMA_DEG = 0.001 # Ultra-low azimuth noise (95% reduction)
LIDAR_ELEVATION_SIGMA_DEG = 0.001 # Ultra-low elevation noise (95% reduction)
LIDAR_RANGE_RATE_SIGMA_M_S = 0.01 # Ultra-low range rate noise (95% reduction)
PROBABILITY_OF_DETECTION_MAX = 0.999 # Near-perfect detection (99.9%)
CLUTTER_RATE_POISSON_LAMBDA = 0.01  # Ultra-minimal clutter for clean learning

# =============================================================================
# LMB FILTER PARAMETERS (SIMPLIFIED FOR SPEED)
# =============================================================================
BIRTH_PROBABILITY = 0.02         # Reduced from 0.05 (very conservative)
SURVIVAL_PROBABILITY = 0.998     # Increased from 0.995 (very stable tracks)
EXISTENCE_THRESHOLD_PRUNING = 0.001  # Reduced from 0.005 (keep fewer tracks)
EXISTENCE_THRESHOLD_EXTRACTION = 0.2  # Reduced from 0.3 (include fewer tracks)
PROCESS_NOISE_Q = 0.2           # Reduced from 0.5 (minimal uncertainty)

# =============================================================================
# GNN & RL PARAMETERS (OPTIMIZED FOR STABLE LEARNING)
# =============================================================================
LEARNING_RATE = 3e-4            # Reduced from 1e-3 (more stable learning)
DISCOUNT_FACTOR_GAMMA = 0.95    # Increased from 0.9 (better long-term planning)
PPO_CLIP_EPSILON = 0.1          # Increased from 0.05 (less conservative updates)
NUM_EPOCHS_PER_UPDATE = 3       # Increased from 2 (more stable updates)
BATCH_SIZE = 32                 # Increased from 16 (more stable gradients)
K_NEAREST_NEIGHBORS = 6         # Increased from 4 (better graph connectivity)
HIDDEN_DIM = 128                # Increased from 64 (more network capacity)
NUM_ATTENTION_HEADS = 2         # Increased from 1 (better attention mechanism)

# =============================================================================
# TRAINING PARAMETERS (AGGRESSIVELY REDUCED)
# =============================================================================
TOTAL_EPISODES = 200            # Reduced from 400 (50% fewer episodes)
SAVE_MODEL_EVERY = 25           # Reduced from 50 (more frequent saves)
EVALUATION_EVERY = 20           # Reduced from 25 (more frequent evaluation)

# =============================================================================
# FILE PATHS
# =============================================================================
# Get the base directory (parent of simulation folder)
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
EARTH_RADIUS_M = 6371000.0  # Earth radius in meters
GRAVITATIONAL_CONSTANT = 3.986004418e14  # Earth's gravitational parameter (m³/s²)
ORBITAL_PERIOD_LEO = 90.0  # Typical LEO orbital period in minutes

# =============================================================================
# OSPA PARAMETERS (RELAXED FOR EASIER LEARNING)
# =============================================================================
OSPA_CUTOFF_C = 5000000.0        # Temporarily increased to 5000km for testing
OSPA_ORDER_P = 1.0              # Keep same

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"
SEED = 42  # Random seed for reproducibility 