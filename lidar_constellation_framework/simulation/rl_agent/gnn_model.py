"""
Graph Neural Network Model for LiDAR Constellation Sensor Tasking

This module implements the Actor-Critic GNN architecture with a shared backbone
for processing graph representations of the debris environment and outputting
sensor tasking decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
import config


class ActorCriticGNN(nn.Module):
    """
    Shared-backbone Actor-Critic Graph Neural Network for sensor tasking.
    
    The architecture consists of:
    1. Shared GNN backbone (3 GAT layers + global pooling)
    2. Actor head (MLP for policy)
    3. Critic head (MLP for value function)
    """
    
    def __init__(self):
        """Initialize the Actor-Critic GNN architecture."""
        super(ActorCriticGNN, self).__init__()
        
        # =============================================================================
        # SHARED GNN BACKBONE
        # =============================================================================
        
        # Layer 2.1 - GATConv 1: 13 -> 64 features, 2 attention heads
        self.gat_conv1 = GATConv(
            in_channels=13,  # Node features: [6D state + 6D covariance diagonal + 1D existence]
            out_channels=64,
            heads=config.NUM_ATTENTION_HEADS,
            dropout=0.1,
            concat=True
        )
        
        # Layer 2.2 - GATConv 2: 128 -> 128 features, 2 attention heads
        # Input: 64 * 2 = 128 features from gat_conv1
        self.gat_conv2 = GATConv(
            in_channels=64 * config.NUM_ATTENTION_HEADS,  # 64 * 2 = 128
            out_channels=128,
            heads=config.NUM_ATTENTION_HEADS,
            dropout=0.1,
            concat=True
        )
        
        # Layer 2.3 - GATConv 3: 256 -> 256 features, 2 attention heads
        # Input: 128 * 2 = 256 features from gat_conv2
        self.gat_conv3 = GATConv(
            in_channels=128 * config.NUM_ATTENTION_HEADS,  # 128 * 2 = 256
            out_channels=256,
            heads=config.NUM_ATTENTION_HEADS,
            dropout=0.1,
            concat=True
        )
        
        # Layer 2.4 - Global Mean Pooling: aggregates node features to global context vector
        # This is handled in the forward method using global_mean_pool function
        
        # =============================================================================
        # ACTOR HEAD (MLP for Policy)
        # =============================================================================
        
        # Layer 3.1 - Fully Connected 1: 512 -> 512
        # Input: 256 * 2 = 512 features from gat_conv3
        self.actor_fc1 = nn.Linear(256 * config.NUM_ATTENTION_HEADS, 512)
        
        # Layer 3.2 - Fully Connected 2: 512 -> 1024
        self.actor_fc2 = nn.Linear(512, 1024)
        
        # Layer 3.3 - Actor Output: 1024 -> 80 (2 outputs per satellite * 40 satellites)
        self.actor_output = nn.Linear(1024, config.NUM_SATELLITES * 2)  # 2 actions per satellite (azimuth, elevation)
        
        # =============================================================================
        # CRITIC HEAD (MLP for Value Function)
        # =============================================================================
        
        # Layer 4.1 - Fully Connected 1: 512 -> 128
        # Input: 256 * 2 = 512 features from gat_conv3
        self.critic_fc1 = nn.Linear(256 * config.NUM_ATTENTION_HEADS, 128)
        
        # Layer 4.2 - Fully Connected 2: 128 -> 64
        self.critic_fc2 = nn.Linear(128, 64)
        
        # Layer 4.3 - Critic Output: 64 -> 1 (single scalar value)
        self.critic_output = nn.Linear(64, 1)
        
        # =============================================================================
        # INITIALIZATION
        # =============================================================================
        
        # Initialize weights using Xavier/Glorot initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, graph_data: Data) -> tuple:
        """
        Forward pass through the Actor-Critic GNN.
        
        Args:
            graph_data: PyTorch Geometric Data object containing:
                - x: Node features tensor (N x 13)
                - edge_index: Edge connectivity tensor (2 x E)
                - batch: Batch assignment tensor (N)
        
        Returns:
            tuple: (action_logits, state_value)
                - action_logits: Tensor of shape (batch_size, 80) with tanh activation
                - state_value: Tensor of shape (batch_size, 1) with linear activation
        """
        # Extract graph components
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        
        # =============================================================================
        # SHARED GNN BACKBONE
        # =============================================================================
        
        # Layer 2.1 - GATConv 1 with ReLU activation
        x = F.relu(self.gat_conv1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Layer 2.2 - GATConv 2 with ReLU activation
        x = F.relu(self.gat_conv2(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Layer 2.3 - GATConv 3 with ReLU activation
        x = F.relu(self.gat_conv3(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Layer 2.4 - Global Mean Pooling to get fixed-size representation
        # This aggregates all node features into a single 256-dimensional vector
        global_context = global_mean_pool(x, batch)  # Shape: (batch_size, 256)
        
        # =============================================================================
        # ACTOR PATH (Policy Network)
        # =============================================================================
        
        # Layer 3.1 - Fully Connected 1 with ReLU activation
        actor_hidden1 = F.relu(self.actor_fc1(global_context))
        
        # Layer 3.2 - Fully Connected 2 with ReLU activation
        actor_hidden2 = F.relu(self.actor_fc2(actor_hidden1))
        
        # Layer 3.3 - Actor Output with tanh activation (normalizes to [-1, 1])
        # Reshape to (batch_size, num_satellites, 2) for azimuth and elevation
        action_logits = torch.tanh(self.actor_output(actor_hidden2))
        action_logits = action_logits.view(-1, config.NUM_SATELLITES, 2)
        
        # =============================================================================
        # CRITIC PATH (Value Network)
        # =============================================================================
        
        # Layer 4.1 - Fully Connected 1 with ReLU activation
        critic_hidden1 = F.relu(self.critic_fc1(global_context))
        
        # Layer 4.2 - Fully Connected 2 with ReLU activation
        critic_hidden2 = F.relu(self.critic_fc2(critic_hidden1))
        
        # Layer 4.3 - Critic Output with linear activation (unbounded value)
        state_value = self.critic_output(critic_hidden2)
        
        return action_logits, state_value

    
    def get_parameter_count(self) -> int:
        """Get the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, filepath: str):
        """Save the model to a file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'num_satellites': config.NUM_SATELLITES,
                'hidden_dim': config.HIDDEN_DIM,
                'num_attention_heads': config.NUM_ATTENTION_HEADS
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load the model from a file."""
        checkpoint = torch.load(filepath, map_location=config.DEVICE)
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('config', {}) 