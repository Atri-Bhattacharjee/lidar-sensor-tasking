"""
Memory-Optimized PPO Agent for LiDAR Constellation Sensor Tasking

This module implements a memory-efficient version of the PPO training algorithm
that uses smaller buffers and more efficient data handling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import config
from .gnn_model import ActorCriticGNN


class MemoryOptimizedPPOMemory:
    """
    Memory-optimized buffer for storing PPO trajectories.
    """
    
    def __init__(self, max_size: int = None):
        """
        Initialize the memory-optimized PPO memory buffer.
        
        Args:
            max_size: Maximum number of transitions to store (uses config if None)
        """
        self.max_size = max_size or getattr(config, 'PPO_MEMORY_SIZE', 5000)
        self.reset()
    
    def reset(self):
        """Reset the memory buffer."""
        self.states = deque(maxlen=self.max_size)
        self.actions = deque(maxlen=self.max_size)
        self.rewards = deque(maxlen=self.max_size)
        self.log_probs = deque(maxlen=self.max_size)
        self.values = deque(maxlen=self.max_size)
        self.dones = deque(maxlen=self.max_size)
    
    def add(self, state, action, reward, log_prob, value, done):
        """
        Add a transition to the memory buffer.
        
        Args:
            state: Graph state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of the action
            value: State value estimate
            done: Whether episode ended
        """
        # Convert to tensors if needed and move to CPU to save GPU memory
        if torch.is_tensor(state):
            state = state.cpu()
        if torch.is_tensor(action):
            action = action.cpu()
        if torch.is_tensor(log_prob):
            log_prob = log_prob.cpu()
        if torch.is_tensor(value):
            value = value.cpu()
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def get_all(self) -> Tuple:
        """
        Get all stored transitions.
        
        Returns:
            Tuple of (states, actions, rewards, log_probs, values, dones)
        """
        return (
            list(self.states),
            list(self.actions),
            list(self.rewards),
            list(self.log_probs),
            list(self.values),
            list(self.dones)
        )
    
    def __len__(self):
        return len(self.states)


class MemoryOptimizedPPOAgent:
    """
    Memory-optimized PPO agent for sensor tasking.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the memory-optimized PPO agent.
        
        Args:
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.device = device or config.DEVICE
        
        # Initialize the Actor-Critic GNN model
        self.model = ActorCriticGNN().to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            eps=1e-5
        )
        
        # Initialize memory buffer with smaller size
        self.memory = MemoryOptimizedPPOMemory()
        
        # Training parameters
        self.batch_size = config.BATCH_SIZE
        self.num_epochs = config.NUM_EPOCHS_PER_UPDATE
        self.clip_epsilon = config.PPO_CLIP_EPSILON
        self.gamma = config.DISCOUNT_FACTOR_GAMMA
        self.gae_lambda = 0.95
        
        # Training statistics
        self.training_stats = {
            'policy_losses': [],
            'value_losses': [],
            'total_losses': [],
            'entropy_losses': []
        }
    
    def select_action(self, state_graph) -> Tuple[np.ndarray, float, float]:
        """
        Select an action using the current policy.
        
        Args:
            state_graph: Graph representation of the current state
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        self.model.eval()
        with torch.no_grad():
            # Move state to device
            if not state_graph.x.is_cuda and self.device == 'cuda':
                state_graph = state_graph.to(self.device)
            
            action_logits, state_value = self.model(state_graph)
            action_dist = torch.distributions.Normal(
                loc=torch.tanh(action_logits),
                scale=0.1
            )
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)
            state_value = state_value.squeeze(-1)
        
        # Move to CPU for storage
        action_np = action.cpu().numpy()
        log_prob_np = log_prob.cpu().numpy()
        value_np = state_value.cpu().numpy()
        
        return action_np, log_prob_np, value_np
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """
        Store a transition in the memory buffer.
        
        Args:
            state: Graph state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of the action
            value: State value estimate
            done: Whether episode ended
        """
        self.memory.add(state, action, reward, log_prob, value, done)
    
    def update(self) -> Optional[Dict[str, float]]:
        """
        Update the policy using PPO.
        
        Returns:
            Dictionary containing training statistics, or None if not enough data
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Get all transitions
        states, actions, rewards, log_probs, values, dones = self.memory.get_all()
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages_and_returns(rewards, values, dones)
        
        # Convert to tensors
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        log_probs_tensor = torch.tensor(np.array(log_probs), dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Training loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        
        # Create batches
        batch_indices = self._create_batches(len(states))
        
        for epoch in range(self.num_epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy_loss = 0.0
            
            for batch_idx in batch_indices:
                # Get batch data
                batch_states = [states[i] for i in batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_log_probs = log_probs_tensor[batch_idx]
                batch_advantages = advantages_tensor[batch_idx]
                batch_returns = returns_tensor[batch_idx]
                
                # Forward pass
                batch_logits, batch_values = self.model(batch_states)
                batch_action_dist = torch.distributions.Normal(
                    loc=torch.tanh(batch_logits),
                    scale=0.1
                )
                batch_new_log_probs = batch_action_dist.log_prob(batch_actions).sum(dim=-1)
                batch_entropy = batch_action_dist.entropy().mean(dim=-1)
                
                # Compute ratios
                ratio = torch.exp(batch_new_log_probs - batch_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(batch_values.squeeze(-1), batch_returns)
                
                # Entropy loss for exploration
                entropy_loss = -batch_entropy.mean()
                
                # Total loss
                total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                
                self.optimizer.step()
                
                # Accumulate losses
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy_loss += entropy_loss.item()
            
            # Average losses over batches
            num_batches = len(batch_indices)
            total_policy_loss += epoch_policy_loss / num_batches
            total_value_loss += epoch_value_loss / num_batches
            total_entropy_loss += epoch_entropy_loss / num_batches
        
        # Average losses over epochs
        avg_policy_loss = total_policy_loss / self.num_epochs
        avg_value_loss = total_value_loss / self.num_epochs
        avg_entropy_loss = total_entropy_loss / self.num_epochs
        avg_total_loss = avg_policy_loss + 0.5 * avg_value_loss + 0.01 * avg_entropy_loss
        
        # Store training statistics
        self.training_stats['policy_losses'].append(avg_policy_loss)
        self.training_stats['value_losses'].append(avg_value_loss)
        self.training_stats['entropy_losses'].append(avg_entropy_loss)
        self.training_stats['total_losses'].append(avg_total_loss)
        
        # Clear memory
        self.memory.reset()
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_total_loss
        }
    
    def _compute_advantages_and_returns(self, rewards: List[float], 
                                      values: List[float], 
                                      dones: List[bool]) -> Tuple[List[float], List[float]]:
        """
        Compute advantages and returns using GAE.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        
        # Compute GAE
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages.tolist(), returns
    
    def _create_batches(self, num_samples: int) -> List[List[int]]:
        """
        Create batches for training.
        
        Args:
            num_samples: Number of samples
            
        Returns:
            List of batch indices
        """
        indices = np.random.permutation(num_samples)
        batches = []
        
        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if len(batch_indices) >= self.batch_size // 2:  # Only use batches that are at least half full
                batches.append(batch_indices)
        
        return batches
    
    def save_model(self, filepath: str):
        """Save the model to a file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load the model from a file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """Get training statistics."""
        return self.training_stats.copy()
    
    def reset_training_stats(self):
        """Reset training statistics."""
        for key in self.training_stats:
            self.training_stats[key] = []
    
    def get_parameter_count(self) -> int:
        """Get the number of parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad) 