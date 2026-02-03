"""
Benchmark: Q-Learning vs Gradient Descent
==========================================
Simulates micro-expression detection to compare learning speed.
"""

import numpy as np
import matplotlib.pyplot as plt
from rl_agent import QLearningThresholdAgent

# Define the old Gradient Agent for comparison (Since rl_agent.py was overwritten)
class GradientAgent:
    """
    Simulates the original SensitivityAgent (Gradient Descent / Heuristic).
    """
    def __init__(self, initial_threshold=2.0, min_threshold=1.5, max_threshold=4.0, learning_rate=0.05):
        self.threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.learning_rate = learning_rate
        self.history = []

    def update(self, result, is_flash):
        if is_flash:
            emotion, confidence = result
            # Decrease threshold (more sensitive) if confidence is low
            # Increase threshold (less sensitive) if confidence is high (False Positive check)
            
            # Simple heuristic from V1 logic:
            # If High Confidence (Real flash likely) -> Stabilize
            # If Low Confidence (Weak flash) -> Lower Threshold (Catch more)
            # If Noise (Neutral) -> Raise Threshold
            
            if emotion == 'neutral':
                # False positive -> Raise threshold
                change = 0.1
            elif confidence > 0.7:
                # Good catch -> Stabilize (Small raise to refine)
                change = 0.01
            else:
                # Weak catch -> Lower threshold
                change = -0.05
        else:
            # Decay (No flash) -> Lower threshold slowly to find signal
            change = -0.001
            
        self.threshold += change
        self.threshold = np.clip(self.threshold, self.min_threshold, self.max_threshold)
        self.history.append(self.threshold)
        return self.threshold

class MicroExpressionSimulator:
    """
    Simulates a face with micro-expressions.
    
    Ground truth: optimal threshold is around 2.2
    """
    def __init__(self, true_optimal=2.2):
        self.true_optimal = true_optimal
        
    def simulate_flash(self, threshold):
        """
        Simulate whether a flash is detected at given threshold.
        Returns: (is_flash, emotion, confidence)
        """
        # Probability of flash depends on threshold
        # Too low: many false positives
        # Too high: miss real expressions
        
        # Distance from optimal
        distance = abs(threshold - self.true_optimal)
        
        # Generate random micro-expression (10% chance per call)
        if np.random.random() < 0.1:
            # Real micro-expression occurred
            
            # Detection probability decreases with distance from optimal
            detect_prob = np.exp(-distance / 0.5)
            
            if np.random.random() < detect_prob:
                # Successfully detected
                # Confidence decreases with distance from optimal
                confidence = max(0.5, 0.95 - distance * 0.3)
                emotion = np.random.choice(['happiness', 'sadness', 'anger', 'fear'])
                return True, emotion, confidence
            else:
                # Missed it (threshold too high)
                return False, None, 0.0
        else:
            # No real expression
            
            # False positive probability increases as threshold decreases
            false_pos_prob = np.exp(-(threshold - 1.5) / 0.3)
            
            if np.random.random() < false_pos_prob:
                # False positive
                return True, 'neutral', 0.3
            else:
                # Correctly didn't trigger
                return False, None, 0.0


def run_simulation(agent_class, agent_name, simulator, n_steps=200):
    """Run simulation for one agent."""
    print(f"\nRunning {agent_name}...")
    
    # Initialize agent
    if agent_name == "Q-Learning":
        agent = agent_class(
            min_threshold=1.5,
            max_threshold=4.0,
            n_states=15,
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.4,
            epsilon_decay=0.995,
            epsilon_min=0.05
        )
    else:
        # Gradient Agent instantiation
        agent = agent_class(
            initial_threshold=2.0,
            min_threshold=1.5,
            max_threshold=4.0,
            learning_rate=0.05
        )
    
    thresholds = []
    rewards = []
    true_positives = 0
    false_positives = 0
    
    for step in range(n_steps):
        # Get current threshold
        if hasattr(agent, 'get_threshold'):
            threshold = agent.get_threshold()
        else:
            threshold = agent.threshold
        
        thresholds.append(threshold)
        
        # Simulate
        is_flash, emotion, confidence = simulator.simulate_flash(threshold)
        
        # Calculate reward for tracking
        if is_flash:
            if emotion == 'neutral':
                reward = -1.0
                false_positives += 1
            elif confidence > 0.6:
                reward = 1.0
                true_positives += 1
            else:
                reward = 0.5
                true_positives += 1
        else:
            reward = 0.0
        
        rewards.append(reward)
        
        # Update agent
        if is_flash:
            result = (emotion, confidence)
            agent.update(result, True)
        else:
            agent.update(None, False)
    
    return {
        'thresholds': np.array(thresholds),
        'rewards': np.array(rewards),
        'cumulative_reward': np.cumsum(rewards),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'final_threshold': thresholds[-1]
    }


def plot_comparison(results_ql, results_gd, true_optimal):
    """Plot comparison charts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    steps = len(results_ql['thresholds'])
    x = np.arange(steps)
    
    # 1. Threshold evolution
    ax = axes[0, 0]
    ax.plot(x, results_ql['thresholds'], label='Q-Learning', linewidth=2, alpha=0.8)
    ax.plot(x, results_gd['thresholds'], label='Gradient Descent', linewidth=2, alpha=0.8)
    ax.axhline(y=true_optimal, color='green', linestyle='--', label='True Optimal', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Threshold')
    ax.set_title('Threshold Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Cumulative reward
    ax = axes[0, 1]
    ax.plot(x, results_ql['cumulative_reward'], label='Q-Learning', linewidth=2, alpha=0.8)
    ax.plot(x, results_gd['cumulative_reward'], label='Gradient Descent', linewidth=2, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward (Higher is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Rolling average reward (last 20 steps)
    ax = axes[1, 0]
    window = 20
    ql_rolling = np.convolve(results_ql['rewards'], np.ones(window)/window, mode='valid')
    gd_rolling = np.convolve(results_gd['rewards'], np.ones(window)/window, mode='valid')
    ax.plot(x[window-1:], ql_rolling, label='Q-Learning', linewidth=2, alpha=0.8)
    ax.plot(x[window-1:], gd_rolling, label='Gradient Descent', linewidth=2, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Average Reward (20-step window)')
    ax.set_title('Learning Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Performance metrics
    ax = axes[1, 1]
    metrics = ['True Positives', 'False Positives', 'Net Score']
    
    ql_tp = results_ql['true_positives']
    ql_fp = results_ql['false_positives']
    gd_tp = results_gd['true_positives']
    gd_fp = results_gd['false_positives']
    
    ql_scores = [ql_tp, ql_fp, ql_tp - ql_fp]
    gd_scores = [gd_tp, gd_fp, gd_tp - gd_fp]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x_pos - width/2, ql_scores, width, label='Q-Learning', alpha=0.8)
    ax.bar(x_pos + width/2, gd_scores, width, label='Gradient Descent', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Count')
    ax.set_title('Detection Performance')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('rl_comparison.png', dpi=150)
    print("\nComparison plot saved to: rl_comparison.png")
    # plt.show() # Disabled for headless checking


def main():
    print("="*60)
    print("Q-LEARNING vs GRADIENT DESCENT BENCHMARK")
    print("="*60)
    print("Simulating micro-expression detection with both methods...")
    print("True optimal threshold: 2.2")
    print("="*60)
    
    # Create simulator
    simulator = MicroExpressionSimulator(true_optimal=2.2)
    
    # Run simulations
    results_ql = run_simulation(QLearningThresholdAgent, "Q-Learning", simulator, n_steps=200)
    results_gd = run_simulation(GradientAgent, "Gradient Descent", simulator, n_steps=200)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print("\nQ-Learning:")
    print(f"  Final threshold: {results_ql['final_threshold']:.3f}")
    print(f"  Distance from optimal: {abs(results_ql['final_threshold'] - 2.2):.3f}")
    print(f"  True positives: {results_ql['true_positives']}")
    print(f"  False positives: {results_ql['false_positives']}")
    print(f"  Total reward: {results_ql['cumulative_reward'][-1]:.1f}")
    
    print("\nGradient Descent:")
    print(f"  Final threshold: {results_gd['final_threshold']:.3f}")
    print(f"  Distance from optimal: {abs(results_gd['final_threshold'] - 2.2):.3f}")
    print(f"  True positives: {results_gd['true_positives']}")
    print(f"  False positives: {results_gd['false_positives']}")
    print(f"  Total reward: {results_gd['cumulative_reward'][-1]:.1f}")
    
    # Calculate convergence time (when threshold stays within 0.2 of optimal)
    def convergence_time(thresholds, optimal, tolerance=0.2):
        for i in range(len(thresholds) - 10):
            if all(abs(thresholds[i:i+10] - optimal) < tolerance):
                # Ensure it stays converged for a while
                return i
        return len(thresholds)
    
    ql_converge = convergence_time(results_ql['thresholds'], 2.2)
    gd_converge = convergence_time(results_gd['thresholds'], 2.2)
    
    print("\n" + "-"*60)
    print("CONVERGENCE ANALYSIS")
    print("-"*60)
    print(f"Q-Learning converged at step: {ql_converge}")
    print(f"Gradient Descent converged at step: {gd_converge}")
    
    if ql_converge < gd_converge:
        speedup = gd_converge / max(1, ql_converge)
        print(f"\n✓ Q-Learning is {speedup:.1f}x FASTER!")
    else:
        print(f"\n✗ Q-Learning was slower in this run (try again - it's stochastic)")
    
    print("="*60)
    
    # Plot comparison
    plot_comparison(results_ql, results_gd, true_optimal=2.2)


if __name__ == "__main__":
    main()
