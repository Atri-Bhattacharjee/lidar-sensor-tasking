# Optimized Configuration for Fast Training with Full Constellation (1-3 hours)
# This configuration uses all 40 satellites while maintaining fast convergence

import os

# =============================================================================
# SIMULATION PARAMETERS (REDUCED FOR FASTER TRAINING)
# =============================================================================
SIMULATION_TIME_STEPS = 50      # Reduced from 100 (50% shorter episodes)
EPISODE_DURATION_SECONDS = 1800  # Reduced from 3600 (30 minutes instead of 1 hour)

# =============================================================================
# CONSTELLATION PARAMETERS (FULL CONSTELLATION)
# =============================================================================
NUM_SATELLITES = 40             # Full constellation (all satellites)
SENSOR_SLEW_RATE_DEG_S = 2.0    # Keep same
SENSOR_FOV_DEG = 15.0           # Increased from 10.0 (easier detection)
SENSOR_MAX_RANGE_M = 1000000    # Keep same

# =============================================================================
# SENSOR NOISE PARAMETERS (REDUCED NOISE FOR EASIER LEARNING)
# =============================================================================
LIDAR_RANGE_SIGMA_M = 5.0       # Reduced from 10.0 (less noise)
LIDAR_AZIMUTH_SIGMA_DEG = 0.05  # Reduced from 0.1 (less noise)
LIDAR_ELEVATION_SIGMA_DEG = 0.05 # Reduced from 0.1 (less noise)
LIDAR_RANGE_RATE_SIGMA_M_S = 0.5 # Reduced from 1.0 (less noise)
PROBABILITY_OF_DETECTION_MAX = 0.98  # Increased from 0.95 (easier detection)
CLUTTER_RATE_POISSON_LAMBDA = 2.0   # Reduced from 5.0 (less clutter)

# =============================================================================
# LMB FILTER PARAMETERS (OPTIMIZED FOR FASTER CONVERGENCE)
# =============================================================================
BIRTH_PROBABILITY = 0.05         # Reduced from 0.1 (more conservative)
SURVIVAL_PROBABILITY = 0.995     # Increased from 0.99 (more stable tracks)
EXISTENCE_THRESHOLD_PRUNING = 0.005  # Reduced from 0.01 (keep more tracks)
EXISTENCE_THRESHOLD_EXTRACTION = 0.3  # Reduced from 0.5 (include more tracks)
PROCESS_NOISE_Q = 0.5           # Reduced from 1.0 (less uncertainty)

# =============================================================================
# GNN & RL PARAMETERS (OPTIMIZED FOR FAST CONVERGENCE WITH FULL CONSTELLATION)
# =============================================================================
LEARNING_RATE = 6e-4            # Increased from 5e-4 (faster learning for full constellation)
DISCOUNT_FACTOR_GAMMA = 0.95    # Reduced from 0.99 (shorter horizon)
PPO_CLIP_EPSILON = 0.1          # Reduced from 0.2 (more conservative updates)
NUM_EPOCHS_PER_UPDATE = 4       # Reduced from 5 (faster updates)
BATCH_SIZE = 32                 # Reduced from 64 (faster processing)
K_NEAREST_NEIGHBORS = 6         # Reduced from 8 (simpler graphs)
HIDDEN_DIM = 128                # Reduced from 256 (smaller network)
NUM_ATTENTION_HEADS = 2         # Reduced from 4 (simpler attention)

# =============================================================================
# TRAINING PARAMETERS (OPTIMIZED FOR FAST COMPLETION WITH FULL CONSTELLATION)
# =============================================================================
TOTAL_EPISODES = 400            # Reduced from 500 (faster completion with full constellation)
SAVE_MODEL_EVERY = 50           # Reduced from 100 (more frequent saves)
EVALUATION_EVERY = 25           # Reduced from 50 (more frequent evaluation)

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
OSPA_CUTOFF_C = 2000.0          # Increased from 1000.0 (more forgiving)
OSPA_ORDER_P = 1.0              # Keep same

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"
SEED = 42  # Random seed for reproducibility 