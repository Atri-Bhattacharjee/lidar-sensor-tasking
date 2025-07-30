# LiDAR Constellation Sensor Tasking Framework

Note: some of the code in this repository was created with the assistance of AI, including Gemini 2.5 Pro by Google and Cursor AI. The AI written portions of the code were checked and debugged by me, and the rest of the code was written by me.

A complete Python-based simulation and training framework for autonomous LiDAR constellation sensor tasking systems. This project implements a reinforcement learning approach using Graph Neural Networks (GNNs) and Proximal Policy Optimization (PPO) to optimize sensor pointing decisions for space debris tracking.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data generation, simulation, and training
- **State-of-the-Art Tracking**: Labeled Multi-Bernoulli (LMB) filter for robust multi-object tracking
- **Graph Neural Networks**: GAT-based architecture for processing dynamic debris environments
- **Reinforcement Learning**: PPO algorithm for learning optimal sensor tasking policies
- **Performance Evaluation**: OSPA (Optimal Subpattern Assignment) distance metric for tracking performance assessment
- **Comprehensive Simulation**: Realistic sensor noise, detection probability, and clutter modeling

## Project Structure

```
lidar_constellation_framework/
├── data_generation/           # Data generation pipeline
│   ├── preprocess_cpe_files.py  # CPE file preprocessing
│   ├── master_scraper.py
│   ├── constellation.csv
│   ├── template.inp
│   └── master.cfg
├── simulation/                # Main simulation framework
│   ├── config.py             # Configuration and hyperparameters
│   ├── main.py               # Main training script
│   ├── environment/          # RL environment components
│   │   ├── __init__.py
│   │   ├── constellation_env.py    # Main RL environment
│   │   ├── perception_layer.py     # Sensor simulation
│   │   └── estimation_layer.py     # LMB filter
│   ├── rl_agent/            # Reinforcement learning components
│   │   ├── __init__.py
│   │   ├── gnn_model.py     # Actor-Critic GNN
│   │   └── ppo_agent.py     # PPO training algorithm
│   └── utils/               # Utility functions
│       ├── __init__.py
│       ├── ospa.py          # OSPA distance calculation
│       └── performance_monitor.py  # Performance monitoring
├── ground_truth_data/        # Ground truth debris catalog
│   └── ground_truth_database.pkl  # Preprocessed ground truth
├── output/                   # Input CPE files directory
│   └── *_cond.cpe           # CPE event log files
├── results/                  # Training outputs
│   ├── models/              # Trained model weights
│   ├── plots/               # Training curves and analysis
│   └── training_logs/       # Training logs
├── requirements.txt          # Python dependencies
├── test_preprocessing.py     # Preprocessing test script
├── demo_performance_monitoring.py  # Performance monitoring demo
└── README.md                # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd lidar_constellation_framework
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch Geometric** (if not already installed):
   ```bash
   # For CUDA support (replace with your CUDA version)
   pip install torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
   
   # For CPU only
   pip install torch-geometric
   ```

## Data Preprocessing

The framework includes a comprehensive data preprocessing pipeline for handling CPE (Conjunction Prediction Event) files:

### CPE File Processing

1. **Place your `.cond.cpe` files in the `output/` directory**
2. **Run the preprocessing script**:
   ```bash
   python data_generation/preprocess_cpe_files.py
   ```

This script:
- Loads and combines all 40 `*_cond.cpe` files
- Identifies unique debris objects based on orbital parameters
- Propagates each object's trajectory for the entire simulation duration
- Creates a time-indexed ground truth database
- Saves the result as `ground_truth_data/ground_truth_database.pkl`

### Testing the Preprocessing

Test the preprocessing pipeline:
```bash
python test_preprocessing.py
```

This will verify that:
- CPE files can be loaded correctly
- Unique objects are identified properly
- Orbit propagation works as expected
- The environment can load the ground truth database

## Quick Start

1. **Process CPE data files (if available)**:
   ```bash
   python data_generation/preprocess_cpe_files.py
   ```
   This creates the ground truth database from your `*_cond.cpe` files in the `output/` directory.

2. **Test the preprocessing system**:
   ```bash
   python test_preprocessing.py
   ```

3. **Run the test suite**:
   ```bash
   python test_framework.py
   ```

4. **Start training**:
   ```bash
   cd simulation
   python main.py
   ```

5. **Monitor training progress**:
   - Training logs will be printed to console
   - Performance plots are saved to `results/plots/`
   - Model checkpoints are saved to `results/models/`

6. **Evaluate trained model**:
   ```python
   from rl_agent.ppo_agent import PPOAgent
   from environment.constellation_env import ConstellationEnv
   
   # Load trained agent
   agent = PPOAgent()
   agent.load_model('results/models/best_model.pth')
   
   # Evaluate
   env = ConstellationEnv()
   # ... evaluation code
   ```

## Configuration

All hyperparameters and settings are centralized in `simulation/config.py`:

- **Simulation Parameters**: Time steps, episode duration
- **Constellation Parameters**: Number of satellites, sensor characteristics
- **Sensor Noise**: Range, azimuth, elevation measurement uncertainties
- **LMB Filter Parameters**: Birth probability, existence thresholds
- **GNN & RL Parameters**: Learning rate, network architecture, training settings

## Architecture Overview

### 1. Environment Layer
- **ConstellationEnv**: Main RL environment managing the simulation loop
- **PerceptionLayer**: Simulates LiDAR sensor measurements with noise and clutter
- **EstimationLayer**: Implements LMB filter for multi-object tracking

### 2. Agent Layer
- **ActorCriticGNN**: Shared-backbone GNN with separate actor and critic heads
- **PPOAgent**: PPO training algorithm with experience replay and advantage estimation

### 3. Graph Representation
- **Nodes**: Debris tracks with 13-dimensional feature vectors
- **Edges**: k-Nearest Neighbors based on 3D spatial proximity
- **Features**: [6D state + 6D covariance diagonal + 1D existence probability]

## Performance Metrics

The framework uses the **OSPA (Optimal Subpattern Assignment)** distance metric to evaluate tracking performance:

- **Localization Error**: Distance between matched ground truth and estimated objects
- **Cardinality Error**: Penalty for missed detections and false alarms
- **Combined Score**: Weighted combination of localization and cardinality errors

## Key Components

### LMB Filter
- **Gaussian Mixture Models**: Represents state uncertainty
- **Unscented Kalman Filter**: Handles nonlinear orbital dynamics
- **Data Association**: Hungarian algorithm for measurement-to-track assignment
- **Track Management**: Birth, death, and pruning of tracks

### GNN Architecture
- **Graph Attention Networks**: 3-layer GAT with 4 attention heads
- **Global Pooling**: Aggregates node features to fixed-size representation
- **Actor-Critic Heads**: Separate MLPs for policy and value functions

### PPO Training
- **Clipped Objective**: Prevents large policy updates
- **Generalized Advantage Estimation**: Efficient advantage calculation
- **Experience Replay**: Stores and reuses trajectories
- **Gradient Clipping**: Stabilizes training

## Advanced Usage

### Custom Ground Truth Data
Replace the simplified ground truth generation in `constellation_env.py` with your own data loader:

```python
def _load_ground_truth_data(self):
    # Load from .cpe file or other format
    pass
```

### Custom Sensor Models
Modify `perception_layer.py` to implement different sensor characteristics:

```python
def _custom_detection_model(self, obj, satellite_pos):
    # Implement your detection probability model
    pass
```

### Custom Reward Functions
Modify the reward calculation in `constellation_env.py`:

```python
def _calculate_reward(self, ground_truth, estimated_state):
    # Implement custom reward function
    pass
```

## Training Tips

1. **Start Small**: Begin with fewer satellites and objects for faster iteration
2. **Monitor OSPA**: Focus on minimizing OSPA distance rather than just episode reward
3. **Adjust Learning Rate**: Start with default and reduce if training is unstable
4. **Use GPU**: Enable CUDA for significantly faster training
5. **Save Checkpoints**: Regular model saving prevents loss of progress

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and paths are correct
2. **Memory Issues**: Reduce batch size or number of satellites
3. **Training Instability**: Reduce learning rate or increase gradient clipping
4. **Poor Performance**: Check ground truth data quality and sensor parameters

### Debug Mode
Enable debug logging by modifying `config.py`:

```python
DEBUG = True
VERBOSE = True
```

## References

- **LMB Filter**: Vo, B.-N., & Ma, W.-K. (2006). The Gaussian mixture probability hypothesis density filter.
- **OSPA Metric**: Schuhmacher, D., Vo, B.-T., & Vo, B.-N. (2008). A consistent metric for performance evaluation of multi-object filters.
- **PPO**: Schulman, J., et al. (2017). Proximal policy optimization algorithms.
- **Graph Attention Networks**: Veličković, P., et al. (2018). Graph attention networks.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## Acknowledgments

- PyTorch Geometric team for the excellent GNN library
- OpenAI Gym for the RL environment framework
- The space debris tracking community for inspiration and feedback

---

For questions, issues, or contributions, please open an issue or contact the maintainers. 
