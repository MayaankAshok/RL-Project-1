import numpy as np
from Environment import MachineReplacementEnv
from typing import List, Tuple
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import uuid

class QLearningAgent:
    def __init__(
        self,
        env: MachineReplacementEnv,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        ema_decay: float = 0.99  # EMA decay parameter
    ):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.ema_decay = ema_decay
        
        # Initialize Q-table
        self.q_table = np.zeros((env.n_states, len(env.action_space)))
        self.ema_q_table = np.zeros((env.n_states, len(env.action_space)))  # EMA Q-table
        
        # For tracking metrics
        self.episode_rewards = []
        self.cumulative_rewards = []
        self.regrets = []
        self.cumulative_regrets = []
        self.episode_histories = []  # Store episode trajectories
        self.episode_values = []
        
        # Add run identifier
        self.run_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def get_action(self, state: int) -> int:
        """Select action using ε-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.action_space)
        return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        """Update Q-value using TD learning"""
        best_next_value = np.max(self.q_table[next_state])
        current_q = self.q_table[state, action]
        
        # Q-learning update rule
        new_q = current_q + self.lr * (
            reward + self.gamma * best_next_value - current_q
        )
        self.q_table[state, action] = new_q
        
        # Update EMA Q-table
        self.ema_q_table[state, action] = (
            self.ema_decay * self.ema_q_table[state, action] + 
            (1 - self.ema_decay) * new_q
        )
    
    def compute_optimal_value(self, start_state: int) -> float:
        """Compute V*(s) for the current Q-table"""
        state = start_state
        total_reward = 0
        discount = 1.0
        
        for _ in range(self.env.max_steps):
            action = np.argmax(self.q_table[state])
            next_state, reward, done, _ = self.env.step(action)
            total_reward += discount * reward
            discount *= self.gamma
            state = next_state
            if done:
                break
                
        self.env.reset()  # Reset environment after computation
        return total_reward
    
    def compute_value_from_qtable(self, start_state: int, q_table: np.ndarray) -> float:
        """Compute value of a state using given Q-table"""
        return np.max(q_table[start_state])
    def compute_regrets(self):
        """Compute regrets using final Q-table as optimal policy"""
        # Reset metrics
        self.regrets = []
        self.cumulative_regrets = []
        cumulative_regret = 0
        
        # Compute optimal value for initial state using final Q-table
        optimal_value = np.average([self.compute_optimal_value(0) for _ in range(10000)])
        
        # Compute regrets for all episodes
        for episode_reward in self.episode_rewards:
            regret = optimal_value - episode_reward
            self.regrets.append(regret)
            cumulative_regret += regret
            self.cumulative_regrets.append(cumulative_regret)

    def train(self, n_episodes: int):
        """Train the agent"""
        # Initialize EMA Q-table with current Q-table
        self.ema_q_table = self.q_table.copy()
        
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            discount = 1.0
            discounted_reward = 0
            
            # Run episode
            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Update Q-values
                self.update(state, action, reward, next_state)
                
                episode_reward += reward
                discounted_reward += discount * reward
                discount *= self.gamma
                state = next_state
                
                if done:
                    break
            
            # Decay exploration rate
            self.epsilon = max(self.min_epsilon, 
                             self.epsilon * self.epsilon_decay)
            
            # Track metrics
            # self.episode_rewards.append(discounted_reward)
            self.episode_rewards.append(discounted_reward)
            self.episode_values.append(self.compute_value_from_qtable(0, self.q_table))
            self.cumulative_rewards.append(sum(self.episode_rewards))
            
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{n_episodes}")
                print(f"Max Q diff (EMA vs Current): "
                      f"{np.max(np.abs(self.ema_q_table - self.q_table)):.6f}")
        
        # Use EMA Q-table as final Q-table for policy and value computation
        self.q_table = self.ema_q_table.copy()
        
        # Compute regrets after training is complete
        self.compute_regrets()
        
        # Save results
        self.save_results()

    def save_results(self):
        """Save experiment results and parameters to JSON"""
        # Create logs directory if it doesn't exist
        log_dir = "q-learning_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Prepare data to save
        experiment_data = {
            "run_id": self.run_id,
            "timestamp": self.start_time,
            "algorithm": "Q-Learning",
            "hyperparameters": {
                "learning_rate": self.lr,
                "discount_factor": self.gamma,
                "initial_epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "min_epsilon": self.min_epsilon,
                "ema_decay": self.ema_decay,
            },
            "environment_params": {
                "n_states": self.env.n_states,
                "replacement_factor": self.env.K / self.env.h(self.env.n_states - 1),
                "p": self.env.p,
                "max_steps": self.env.max_steps
            },
            "results": {
                "episode_rewards": self.episode_rewards,
                "cumulative_rewards": self.cumulative_rewards,
                "regrets": self.regrets,
                "cumulative_regrets": self.cumulative_regrets,
                "final_q_table": self.q_table.tolist(),
                "final_ema_q_table": self.ema_q_table.tolist()
            }
        }
        
        # Save to file
        filename = f"log_{self.start_time}_{self.run_id}.json"
        filepath = os.path.join(log_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        print(f"Results saved to {filepath}")

    def moving_average(self, data: list, window: int = 50) -> np.ndarray:
        """Compute moving average of data with given window size"""
        weights = np.ones(window) / window
        return np.convolve(data, weights, mode='valid')

    def plot_metrics(self):
        """Plot training metrics with moving averages"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        
        # Plot episode rewards
        ax1.plot(self.episode_rewards, alpha=0.6, label='Raw')
        if len(self.episode_rewards) > 50:  # Only plot MA if enough data
            ma_rewards = self.moving_average(self.episode_rewards)
            ax1.plot(range(49, len(self.episode_rewards)), 
                    ma_rewards, 'r', label='50-ep Moving Avg')
        ax1.set_title('Discounted Rewards per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        
        # Plot cumulative rewards
        ax2.plot(self.cumulative_rewards)
        ax2.set_title('Cumulative Rewards')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cumulative Reward')
        
        # Plot regrets with y=0 line
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.plot(self.regrets, alpha=0.6, label='Raw')
        if len(self.regrets) > 50:
            ma_regrets = self.moving_average(self.regrets)
            ax3.plot(range(49, len(self.regrets)), 
                    ma_regrets, 'r', label='50-ep Moving Avg')
        ax3.set_title('Instantaneous Regret')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Regret')
        ax3.legend()
        
        # Plot cumulative regrets
        ax4.axhline(y=max(self.cumulative_regrets), color='k', linestyle='--', alpha=0.3)
        ax4.plot(self.cumulative_regrets)
        ax4.set_title('Cumulative Regret')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Cumulative Regret')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    env = MachineReplacementEnv()
    agent = QLearningAgent(env)
    agent.train(n_episodes=5000)
    agent.plot_metrics()
