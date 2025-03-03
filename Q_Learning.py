import numpy as np
from Environment import MachineReplacementEnv
from typing import List, Tuple
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(
        self,
        env: MachineReplacementEnv,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01
    ):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize Q-table
        self.q_table = np.zeros((env.n_states, len(env.action_space)))
        
        # For tracking metrics
        self.episode_rewards = []
        self.cumulative_rewards = []
        self.regrets = []
        self.cumulative_regrets = []
        
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
        self.q_table[state, action] = current_q + self.lr * (
            reward + self.gamma * best_next_value - current_q
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
    
    def train(self, n_episodes: int):
        """Train the agent"""
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
            self.episode_rewards.append(discounted_reward)
            self.cumulative_rewards.append(sum(self.episode_rewards))
            
            # Compute regret
            optimal_value = self.compute_optimal_value(0)
            regret = optimal_value - discounted_reward
            self.regrets.append(regret)
            self.cumulative_regrets.append(sum(self.regrets))
            
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{n_episodes}")
    
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
        
        # Plot regrets
        ax3.plot(self.regrets, alpha=0.6, label='Raw')
        if len(self.regrets) > 50:  # Only plot MA if enough data
            ma_regrets = self.moving_average(self .regrets)
            ax3.plot(range(49, len(self.regrets)), 
                    ma_regrets, 'r', label='50-ep Moving Avg')
        ax3.set_title('Instantaneous Regret')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Regret')
        ax3.legend()
        
        # Plot cumulative regrets
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
