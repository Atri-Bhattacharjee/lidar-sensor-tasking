# Memory-Optimized Configuration for LiDAR Constellation Framework
# This configuration prioritizes low memory usage while maintaining training effectiveness

import os

# =============================================================================
# SIMULATION PARAMETERS (MEMORY OPTIMIZED)
# =============================================================================
SIMULATION_TIME_STEPS = 30      # Reduced from 50 (40% shorter episodes)
EPISODE_DURATION_SECONDS = 1200  # Reduced from 1800 (20 minutes instead of 30)

# =============================================================================
# CONSTELLATION PARAMETERS (FULL CONSTELLATION)
# =============================================================================
NUM_SATELLITES = 40             # Full constellation (all satellites from .dia files)
SENSOR_SLEW_RATE_DEG_S = 2.0    # Keep same
SENSOR_FOV_DEG = 20.0           # Keep same
SENSOR_MAX_RANGE_M = 1000000    # Keep same

# =============================================================================
# SENSOR NOISE PARAMETERS (KEEP OPTIMIZED)
# =============================================================================
LIDAR_RANGE_SIGMA_M = 5.0       # Keep optimized
LIDAR_AZIMUTH_SIGMA_DEG = 0.05  # Keep optimized
LIDAR_ELEVATION_SIGMA_DEG = 0.05 # Keep optimized
LIDAR_RANGE_RATE_SIGMA_M_S = 0.5 # Keep optimized
PROBABILITY_OF_DETECTION_MAX = 0.98  # Keep optimized
CLUTTER_RATE_POISSON_LAMBDA = 2.0   # Keep optimized

# =============================================================================
# LMB FILTER PARAMETERS (MEMORY OPTIMIZED)
# =============================================================================
BIRTH_PROBABILITY = 0.03         # Reduced from 0.05 (fewer tracks)
SURVIVAL_PROBABILITY = 0.995     # Keep optimized
EXISTENCE_THRESHOLD_PRUNING = 0.01  # Increased from 0.005 (prune more aggressively)
EXISTENCE_THRESHOLD_EXTRACTION = 0.4  # Increased from 0.3 (fewer tracks to GNN)
PROCESS_NOISE_Q = 0.5           # Keep optimized

# =============================================================================
# GNN & RL PARAMETERS (MEMORY OPTIMIZED)
# =============================================================================
LEARNING_RATE = 5e-4            # Keep optimized
DISCOUNT_FACTOR_GAMMA = 0.95    # Keep optimized
PPO_CLIP_EPSILON = 0.1          # Keep optimized
NUM_EPOCHS_PER_UPDATE = 3       # Reduced from 5 (faster updates)
BATCH_SIZE = 16                 # Reduced from 32 (50% smaller batches)
K_NEAREST_NEIGHBORS = 4         # Reduced from 6 (simpler graphs)
HIDDEN_DIM = 64                 # Reduced from 128 (50% smaller network)
NUM_ATTENTION_HEADS = 2         # Keep optimized

# =============================================================================
# TRAINING PARAMETERS (MEMORY OPTIMIZED)
# =============================================================================
TOTAL_EPISODES = 300            # Reduced from 500 (40% fewer episodes)
SAVE_MODEL_EVERY = 50           # Keep same
EVALUATION_EVERY = 25           # Keep same

# =============================================================================
# MEMORY OPTIMIZATION PARAMETERS (NEW)
# =============================================================================
PPO_MEMORY_SIZE = 5000          # Reduced from 10000 (50% smaller buffer)
MAX_GRAPH_NODES = 200           # Maximum nodes in graph (prevents large graphs)
MAX_TRACKS_PER_TIMESTEP = 50    # Maximum tracks to process per timestep
GROUND_TRUTH_CHUNK_SIZE = 100   # Load ground truth in chunks
PERFORMANCE_MONITOR_MAX_EPISODES = 100  # Limit stored metrics

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
# OSPA PARAMETERS (KEEP OPTIMIZED)
# =============================================================================
OSPA_CUTOFF_C = 2000.0          # Keep optimized
OSPA_ORDER_P = 1.0              # Keep same

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"
SEED = 42  # Random seed for reproducibility 