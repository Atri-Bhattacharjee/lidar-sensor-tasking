# Configuration file for LiDAR Constellation Sensor Tasking Framework
# Centralizes all hyperparameters, simulation settings, file paths, and physical constants

import os

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
SIMULATION_TIME_STEPS = 100
EPISODE_DURATION_SECONDS = 3600  # 1 hour simulation

# =============================================================================
# CONSTELLATION PARAMETERS
# =============================================================================
NUM_SATELLITES = 40
SENSOR_SLEW_RATE_DEG_S = 2.0  # Maximum slew rate in degrees per second
SENSOR_FOV_DEG = 10.0  # Field of view in degrees
SENSOR_MAX_RANGE_M = 1000000  # Maximum detection range in meters

# =============================================================================
# SENSOR NOISE PARAMETERS
# =============================================================================
LIDAR_RANGE_SIGMA_M = 10.0  # Range measurement noise standard deviation
LIDAR_AZIMUTH_SIGMA_DEG = 0.1  # Azimuth measurement noise standard deviation
LIDAR_ELEVATION_SIGMA_DEG = 0.1  # Elevation measurement noise standard deviation
LIDAR_RANGE_RATE_SIGMA_M_S = 1.0  # Range rate measurement noise standard deviation
PROBABILITY_OF_DETECTION_MAX = 0.95  # Maximum probability of detection
CLUTTER_RATE_POISSON_LAMBDA = 5.0  # Average number of clutter measurements per sensor per timestep

# =============================================================================
# LMB FILTER PARAMETERS
# =============================================================================
BIRTH_PROBABILITY = 0.1  # Probability of creating a new track from unassociated measurement
SURVIVAL_PROBABILITY = 0.99  # Probability of track survival between timesteps
EXISTENCE_THRESHOLD_PRUNING = 0.01  # Threshold for pruning tracks
EXISTENCE_THRESHOLD_EXTRACTION = 0.5  # Threshold for extracting tracks for GNN
PROCESS_NOISE_Q = 1.0  # Process noise covariance scaling factor

# =============================================================================
# GNN & RL PARAMETERS
# =============================================================================
LEARNING_RATE = 3e-4
DISCOUNT_FACTOR_GAMMA = 0.99
PPO_CLIP_EPSILON = 0.2
NUM_EPOCHS_PER_UPDATE = 10
BATCH_SIZE = 64
K_NEAREST_NEIGHBORS = 8  # For graph construction
HIDDEN_DIM = 256
NUM_ATTENTION_HEADS = 4

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
TOTAL_EPISODES = 1000
SAVE_MODEL_EVERY = 100  # Save model every N episodes
EVALUATION_EVERY = 50  # Evaluate performance every N episodes

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
# OSPA PARAMETERS
# =============================================================================
OSPA_CUTOFF_C = 1000.0  # OSPA cutoff distance in meters
OSPA_ORDER_P = 1.0  # OSPA order parameter

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"
SEED = 42  # Random seed for reproducibility 