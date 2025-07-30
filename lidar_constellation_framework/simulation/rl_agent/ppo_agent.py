"""
Proximal Policy Optimization (PPO) Agent for LiDAR Constellation Sensor Tasking

This module implements the PPO training algorithm for the GNN-based sensor tasking agent.
It handles experience collection, advantage calculation, and policy updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import config
from .gnn_model import ActorCriticGNN


class PPOMemory:
    """
    Memory buffer for storing PPO trajectories.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize the PPO memory buffer.
        
        Args:
            max_size: Maximum number of transitions to store
        """
        self.max_size = max_size
        self.reset()
    
    def reset(self):
        """Reset the memory buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
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
            self.states,
            self.actions,
            self.rewards,
            self.log_probs,
            self.values,
            self.dones
        )
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    Proximal Policy Optimization agent for sensor tasking.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the PPO agent.
        
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
        
        # Initialize memory buffer (reduced for speed)
        self.memory = PPOMemory(max_size=5000)  # Reduced from 10000
        
        # Training parameters
        self.clip_epsilon = config.PPO_CLIP_EPSILON
        self.gamma = config.DISCOUNT_FACTOR_GAMMA
        self.num_epochs = config.NUM_EPOCHS_PER_UPDATE
        self.batch_size = config.BATCH_SIZE
        
        # Value function loss coefficient
        self.value_coef = 0.5
        
        # Entropy coefficient for exploration
        self.entropy_coef = 0.01
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'total_losses': []
        }
    
    def select_action(self, state_graph) -> Tuple[np.ndarray, float, float]:
        """
        Select an action using the current policy.
        
        Args:
            state_graph: Graph representation of the current state
        
        Returns:
            Tuple of (action, log_probability, state_value)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass to get action logits and value
            action_logits, state_value = self.model(state_graph)
            
            # action_logits shape: (batch_size, num_satellites, 2)
            # state_value shape: (batch_size, 1)
            
            # For single state, batch_size should be 1
            if action_logits.dim() == 3:
                # Remove batch dimension for single state
                action_logits = action_logits.squeeze(0)  # Shape: (num_satellites, 2)
                state_value = state_value.squeeze(0)      # Shape: (1,)
            
            # Create action distribution from logits
            action_dist = torch.distributions.Normal(
                loc=action_logits,  # Already has tanh activation from model
                scale=0.1  # Fixed standard deviation for exploration
            )
            
            # Sample action
            action = action_dist.sample()
            
            # Calculate log probability (sum across satellite and action dimensions)
            log_prob = action_dist.log_prob(action).sum(dim=(-1, -2))  # Sum over satellites and actions
            
            # Ensure state_value is the right shape
            state_value = state_value.squeeze(-1)
        
        # Convert to numpy for environment
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
    
    def update(self) -> Dict[str, float]:
        """
        Update the policy using PPO.
        
        Returns:
            Dictionary containing training statistics
        """
        if len(self.memory) == 0:
            return {}
        
        # Get all transitions from memory
        states, actions, rewards, old_log_probs, old_values, dones = self.memory.get_all()
        
        # Convert to tensors with proper shapes
        actions_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        old_values_tensor = torch.FloatTensor(np.array(old_values)).to(self.device)
        
        # Calculate advantages and returns
        advantages, returns = self._compute_advantages_and_returns(
            rewards, old_values, dones
        )
        
        advantages_tensor = torch.FloatTensor(np.array(advantages)).to(self.device)
        returns_tensor = torch.FloatTensor(np.array(returns)).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Training statistics
        epoch_stats = {
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'total_losses': []
        }
        
        # Multiple epochs of updates
        for epoch in range(self.num_epochs):
            # Create batches
            batch_indices = self._create_batches(len(states))
            
            for batch_idx in batch_indices:
                # Get batch data
                batch_states = [states[i] for i in batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_old_log_probs = old_log_probs_tensor[batch_idx]
                batch_advantages = advantages_tensor[batch_idx]
                batch_returns = returns_tensor[batch_idx]
                
                # Process each state individually (simplified for now)
                batch_logits_list = []
                batch_values_list = []
                
                for state in batch_states:
                    logits, values = self.model(state)
                    batch_logits_list.append(logits)
                    batch_values_list.append(values)
                
                # Stack the results and ensure proper shapes
                batch_logits = torch.stack(batch_logits_list)
                batch_values = torch.stack(batch_values_list)
                # Ensure values have shape (batch_size, 1)
                if batch_values.dim() == 3:
                    batch_values = batch_values.squeeze(-1)
                
                # Create action distribution from logits
                batch_action_dist = torch.distributions.Normal(
                    loc=batch_logits,  # Already has tanh activation from model
                    scale=0.1
                )
                
                # Ensure batch_actions has the right shape for comparison
                if batch_actions.dim() == 2:
                    # If actions are (batch_size, num_satellites*2), reshape to (batch_size, num_satellites, 2)
                    batch_actions = batch_actions.view(batch_actions.shape[0], -1, 2)
                
                # Calculate log probabilities and entropy for the stored actions
                batch_log_probs = batch_action_dist.log_prob(batch_actions).sum(dim=(-1, -2))  # Sum over satellites and actions
                batch_entropy = batch_action_dist.entropy().mean(dim=(-1, -2))  # Mean over satellites and actions
                
                # Calculate policy loss (PPO clipped objective)
                ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss with proper shapes
                # Ensure both tensors have the same shape: (batch_size, 1)
                if batch_values.dim() == 3:
                    batch_values = batch_values.squeeze(-1)  # Remove extra dimension
                if batch_returns.dim() == 1:
                    batch_returns = batch_returns.unsqueeze(-1)  # Add dimension if needed
                value_loss = nn.MSELoss()(batch_values, batch_returns)
                
                # Calculate entropy loss (for exploration)
                entropy_loss = -batch_entropy.mean()
                
                # Total loss
                total_loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping (reduced for more stability)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.3)
                
                self.optimizer.step()
                
                # Store statistics
                epoch_stats['policy_losses'].append(policy_loss.item())
                epoch_stats['value_losses'].append(value_loss.item())
                epoch_stats['entropy_losses'].append(entropy_loss.item())
                epoch_stats['total_losses'].append(total_loss.item())
        
        # Update training statistics
        for key in epoch_stats:
            if epoch_stats[key]:
                self.training_stats[key].append(np.mean(epoch_stats[key]))
        
        # Clear memory
        self.memory.reset()
        
        # Return average statistics for this update
        return {
            'policy_loss': np.mean(epoch_stats['policy_losses']),
            'value_loss': np.mean(epoch_stats['value_losses']),
            'entropy_loss': np.mean(epoch_stats['entropy_losses']),
            'total_loss': np.mean(epoch_stats['total_losses'])
        }
    
    def _compute_advantages_and_returns(self, rewards: List[float], 
                                      values: List[float], 
                                      dones: List[bool]) -> Tuple[List[float], List[float]]:
        """
        Compute advantages and returns using GAE (Generalized Advantage Estimation).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
        
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        
        # GAE parameters
        gae_lambda = 0.95
        
        # Compute returns and advantages in reverse order
        next_value = 0.0
        next_advantage = 0.0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value = 0.0
                next_advantage = 0.0
            
            # Compute TD error
            delta = rewards[i] + self.gamma * next_value - values[i]
            
            # Compute advantage using GAE
            advantage = delta + self.gamma * gae_lambda * next_advantage
            
            # Compute return
            return_val = advantage + values[i]
            
            advantages.insert(0, advantage)
            returns.insert(0, return_val)
            
            next_value = values[i]
            next_advantage = advantage
        
        return advantages, returns
    
    def _create_batches(self, num_samples: int) -> List[List[int]]:
        """
        Create batches for training.
        
        Args:
            num_samples: Number of samples to batch
        
        Returns:
            List of batch indices
        """
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        batches = []
        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batches.append(batch_indices)
        
        return batches
    
    def save_model(self, filepath: str):
        """
        Save the model and training state.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'config': {
                'learning_rate': config.LEARNING_RATE,
                'gamma': config.DISCOUNT_FACTOR_GAMMA,
                'clip_epsilon': config.PPO_CLIP_EPSILON,
                'num_epochs': config.NUM_EPOCHS_PER_UPDATE,
                'batch_size': config.BATCH_SIZE
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """
        Load the model and training state.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        return checkpoint.get('config', {})
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """
        Get training statistics.
        
        Returns:
            Dictionary containing training statistics
        """
        return self.training_stats.copy()
    
    def reset_training_stats(self):
        """Reset training statistics."""
        for key in self.training_stats:
            self.training_stats[key] = []
    
    def get_parameter_count(self) -> int:
        """
        Get the total number of trainable parameters.
        
        Returns:
            Number of parameters
        """
        return self.model.get_parameter_count() 