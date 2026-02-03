"""
Q-Learning Agent for Micro-Expression Sensitivity Tuning
=========================================================

This agent learns the optimal sensitivity threshold through experience.
Uses Q-Learning with:
- Epsilon-greedy exploration
- Experience replay for sample efficiency
- Adaptive learning rate
- State discretization for tractability

Author: Enhanced from original gradient descent approach
"""

import numpy as np
import json
import os
from collections import deque
import random


class QLearningThresholdAgent:
    """
    Q-Learning agent for adaptive threshold tuning.
    
    State Space: Discretized threshold values [1.5, 4.0]
    Action Space: {-0.3, -0.2, -0.1, 0.0, +0.1, +0.2, +0.3}
    Reward: Based on emotion classifier confidence and accuracy
    """
    
    def __init__(
        self,
        min_threshold: float = 1.5,
        max_threshold: float = 4.0,
        n_states: int = 15,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.4,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            min_threshold: Minimum allowed threshold (higher sensitivity)
            max_threshold: Maximum allowed threshold (lower sensitivity)
            n_states: Number of discrete threshold states
            learning_rate: Q-learning alpha (how fast to learn)
            discount_factor: Q-learning gamma (how much to value future rewards)
            epsilon: Initial exploration rate
            epsilon_decay: How fast to reduce exploration
            epsilon_min: Minimum exploration rate
        """
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.n_states = n_states
        
        # Create discrete state space
        self.states = np.linspace(min_threshold, max_threshold, n_states)
        
        # Action space: threshold adjustments
        self.actions = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
        self.n_actions = len(self.actions)
        
        # Q-Table: Q[state_idx][action_idx] = expected reward
        self.Q = np.zeros((n_states, self.n_actions))
        
        # Hyperparameters
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Current state
        self.current_state_idx = n_states // 2  # Start in middle
        self.current_threshold = self.states[self.current_state_idx]
        
        # Experience replay buffer
        self.memory = deque(maxlen=200)  # Store last 200 experiences
        self.batch_size = 10
        
        # Statistics for monitoring
        self.episode_count = 0
        self.total_reward = 0.0
        self.recent_rewards = deque(maxlen=50)
        self.action_counts = np.zeros(self.n_actions)  # Track action usage
        
    def _state_to_idx(self, threshold: float) -> int:
        """Convert continuous threshold to discrete state index."""
        idx = np.argmin(np.abs(self.states - threshold))
        return np.clip(idx, 0, self.n_states - 1)
    
    def _idx_to_state(self, idx: int) -> float:
        """Convert state index to threshold value."""
        return self.states[idx]
    
    def get_threshold(self) -> float:
        """
        Get current threshold for flash detector.
        
        Returns:
            Current threshold value
        """
        return self.current_threshold
    
    def select_action(self, state_idx: int, force_exploit: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        
        Args:
            state_idx: Current state index
            force_exploit: If True, always choose best action (no exploration)
            
        Returns:
            Action index
        """
        # Exploration: random action
        if not force_exploit and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        # Exploitation: best known action
        return np.argmax(self.Q[state_idx])
    
    def _calculate_reward(self, classifier_result, is_flash: bool) -> float:
        """
        Calculate reward based on flash detection outcome.
        
        Reward structure:
        - True Positive (flash + confident emotion): +2.0
        - True Positive (flash + weak emotion): +0.5
        - False Positive (flash + neutral): -2.0
        - No flash: 0.0 (neutral)
        
        Args:
            classifier_result: EmotionResult or tuple (emotion, confidence)
            is_flash: Whether flash was detected
            
        Returns:
            Reward value
        """
        if not is_flash:
            return 0.0  # No event, no reward/penalty
        
        if classifier_result is None:
            return -0.5  # Flash but no classification (technical error)
        
        # Extract emotion and confidence
        if hasattr(classifier_result, 'confidence'):
            emotion = classifier_result.emotion
            confidence = classifier_result.confidence
        else:
            emotion, confidence = classifier_result
        
        # Reward based on detection quality (STARVATION MODE)
        if emotion == 'neutral':
            # False positive: Catastrophic penalty.
            reward = -10.0
        elif confidence > 0.90:
            # Perfection: Only a small reward. Validation is scarce.
            reward = 1.0
        elif confidence > 0.75:
            # Acceptable: Almost negligible reward.
            reward = 0.1
        else:
            # Uncertain signal: Treated as failure.
            reward = -5.0
        
        return reward
    
    def update(self, classifier_result, is_flash: bool) -> float:
        """
        Update Q-function based on experience and return new threshold.
        
        This is the main learning method. Called after each flash event.
        
        Args:
            classifier_result: Result from emotion classifier
            is_flash: Whether a flash was detected
            
        Returns:
            New threshold value to use
        """
        # Calculate reward
        reward = self._calculate_reward(classifier_result, is_flash)
        
        # Store experience only if there was a flash (otherwise no learning signal)
        if is_flash:
            # Select next action
            action_idx = self.select_action(self.current_state_idx)
            self.action_counts[action_idx] += 1
            
            # Calculate next state
            action = self.actions[action_idx]
            next_threshold = np.clip(
                self.current_threshold + action,
                self.min_threshold,
                self.max_threshold
            )
            next_state_idx = self._state_to_idx(next_threshold)
            
            # Store experience (s, a, r, s')
            experience = (
                self.current_state_idx,
                action_idx,
                reward,
                next_state_idx
            )
            self.memory.append(experience)
            
            # Update statistics
            self.episode_count += 1
            self.total_reward += reward
            self.recent_rewards.append(reward)
            
            # Q-Learning update with experience replay
            self._replay_and_learn()
            
            # Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Update current state
            self.current_state_idx = next_state_idx
            self.current_threshold = next_threshold
            
            return self.current_threshold
        
        # No flash: optionally drift towards more sensitivity (lower threshold)
        # This prevents getting "stuck" at high thresholds
        if self.episode_count > 50 and random.random() < 0.01:
            self.current_threshold = max(
                self.min_threshold,
                self.current_threshold - 0.02
            )
            self.current_state_idx = self._state_to_idx(self.current_threshold)
        
        return self.current_threshold
    
    def _replay_and_learn(self):
        """
        Experience replay: learn from random past experiences.
        
        This improves sample efficiency by learning multiple times
        from each experience.
        """
        if len(self.memory) < self.batch_size:
            # Not enough experiences yet, learn from all
            batch = list(self.memory)
        else:
            # Sample random batch
            batch = random.sample(self.memory, self.batch_size)
        
        # Q-Learning update for each experience
        for (state_idx, action_idx, reward, next_state_idx) in batch:
            # Current Q-value
            current_q = self.Q[state_idx, action_idx]
            
            # Maximum Q-value for next state (Bellman equation)
            max_next_q = np.max(self.Q[next_state_idx])
            
            # TD-target: r + γ * max Q(s', a')
            target_q = reward + self.gamma * max_next_q
            
            # Q-Learning update: Q(s,a) ← Q(s,a) + α[target - Q(s,a)]
            self.Q[state_idx, action_idx] = current_q + self.alpha * (target_q - current_q)
    
    def get_status(self) -> str:
        """
        Get human-readable status string for display.
        
        Returns:
            Status string showing key metrics
        """
        if len(self.recent_rewards) == 0:
            avg_reward = 0.0
        else:
            avg_reward = np.mean(self.recent_rewards)
        
        # Calculate sensitivity (inverse of threshold)
        sensitivity = 4.5 - self.current_threshold
        
        # Determine learning phase
        if self.epsilon > 0.2:
            phase = "EXPLORING"
        elif self.epsilon > 0.1:
            phase = "LEARNING"
        else:
            phase = "EXPLOITING"
        
        return (
            f"Q-LEARN: {self.episode_count} episodes | "
            f"ε={self.epsilon:.3f} ({phase}) | "
            f"Sens={sensitivity:.2f} | "
            f"R̄={avg_reward:.2f}"
        )
    
    def get_best_threshold(self) -> float:
        """
        Get the globally best threshold based on learned Q-values.
        
        Returns:
            Threshold with highest expected value
        """
        # For each state, find best action
        state_values = np.max(self.Q, axis=1)
        
        # Find state with highest value
        best_state_idx = np.argmax(state_values)
        
        return self.states[best_state_idx]
    
    def save_state(self, filepath: str) -> None:
        """
        Save learned Q-table and parameters to file.
        
        Args:
            filepath: Path to JSON file
        """
        state = {
            "Q_table": self.Q.tolist(),
            "current_threshold": float(self.current_threshold),
            "current_state_idx": int(self.current_state_idx),
            "epsilon": float(self.epsilon),
            "episode_count": int(self.episode_count),
            "total_reward": float(self.total_reward),
            "action_counts": self.action_counts.tolist(),
            "recent_rewards": list(self.recent_rewards),
            # Hyperparameters
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "n_states": self.n_states,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"[Q-LEARNING] Brain saved to {filepath}")
        except Exception as e:
            print(f"[Q-LEARNING] Failed to save state: {e}")
    
    def load_state(self, filepath: str) -> bool:
        """
        Load learned Q-table and parameters from file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(filepath):
            print(f"[Q-LEARNING] No saved state found at {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore Q-table
            self.Q = np.array(state["Q_table"])
            
            # Restore current state
            self.current_threshold = state["current_threshold"]
            self.current_state_idx = state["current_state_idx"]
            self.epsilon = state["epsilon"]
            self.episode_count = state["episode_count"]
            self.total_reward = state["total_reward"]
            self.action_counts = np.array(state["action_counts"])
            self.recent_rewards = deque(state["recent_rewards"], maxlen=50)
            
            print(f"[Q-LEARNING] Brain loaded from {filepath}")
            print(f"  Episodes trained: {self.episode_count}")
            print(f"  Current threshold: {self.current_threshold:.2f}")
            print(f"  Exploration rate: {self.epsilon:.3f}")
            print(f"  Average recent reward: {np.mean(self.recent_rewards):.2f}")
            
            return True
            
        except Exception as e:
            print(f"[Q-LEARNING] Failed to load state: {e}")
            return False
    
    def get_policy_summary(self) -> dict:
        """
        Get summary of learned policy for analysis.
        
        Returns:
            Dictionary with policy information
        """
        policy = {}
        
        for state_idx in range(self.n_states):
            threshold = self.states[state_idx]
            best_action_idx = np.argmax(self.Q[state_idx])
            best_action = self.actions[best_action_idx]
            q_value = self.Q[state_idx, best_action_idx]
            
            policy[f"threshold_{threshold:.2f}"] = {
                "best_action": float(best_action),
                "q_value": float(q_value),
                "all_q_values": self.Q[state_idx].tolist()
            }
        
        return policy
    
    def print_policy(self):
        """Print human-readable policy summary."""
        print("\n" + "="*60)
        print("LEARNED POLICY SUMMARY")
        print("="*60)
        print(f"Total Episodes: {self.episode_count}")
        print(f"Exploration Rate: {self.epsilon:.3f}")
        print(f"Average Recent Reward: {np.mean(self.recent_rewards):.2f}")
        print("\nAction Distribution:")
        total_actions = self.action_counts.sum()
        if total_actions > 0:
            for i, count in enumerate(self.action_counts):
                action = self.actions[i]
                pct = (count / total_actions) * 100
                print(f"  Action {action:+.1f}: {int(count):3d} times ({pct:.1f}%)")
        
        print("\nTop 5 States (by Q-value):")
        state_values = np.max(self.Q, axis=1)
        top_states = np.argsort(state_values)[-5:][::-1]
        for rank, state_idx in enumerate(top_states, 1):
            threshold = self.states[state_idx]
            q_value = state_values[state_idx]
            best_action_idx = np.argmax(self.Q[state_idx])
            best_action = self.actions[best_action_idx]
            print(f"  {rank}. Threshold={threshold:.2f}, Q={q_value:.2f}, Best Action={best_action:+.1f}")
        
        print("="*60 + "\n")


# Backward compatibility alias
SensitivityAgent = QLearningThresholdAgent
