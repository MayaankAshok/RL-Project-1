import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Environment import MachineReplacementEnv

class PolicyIterationAgent:
    def __init__(self, env: MachineReplacementEnv, discount_factor: float = 0.95, theta: float = 1e-12):
        self.env = env
        self.gamma = discount_factor
        self.theta = theta  # Convergence threshold

        # Initialize deterministic policy (default: continue for all states)
        self.policy = np.zeros(env.n_states, dtype=int)
        self.value_function = np.zeros(env.n_states)

        print("Policy Iteration Agent initialized")
        print(f"Discount Factor (γ): {self.gamma}")
        print(f"Convergence Threshold (θ): {self.theta}")
        print(f"Initial Policy:\n{self.policy}")
        print(f"Initial Value Function:\n{self.value_function}")

        self.episode_rewards = []
        self.cumulative_rewards = []
        self.regrets = []
        self.cumulative_regrets = []

    def get_action(self, state: int) -> int:
        """Select action using the current policy."""
        return self.policy[state]

    def _policy_evaluation(self):
        """Evaluate the current policy until convergence."""
        delta = float('inf')
        while delta > self.theta:
            delta = 0
            for s in range(self.env.n_states):
                v = self.value_function[s]
                
                # Get action from policy
                a = self.policy[s]
                
                # Calculate new value based on Bellman equation
                new_value = 0
                for next_s in range(self.env.n_states):
                    # Get transition probability
                    prob = self.env.P[a, s, next_s]
                    
                    # Get reward (matching how the environment calculates it)
                    reward = -self.env.h(s) if a == 0 else -self.env.K
                    
                    # Add to expected value
                    new_value += prob * (reward + self.gamma * self.value_function[next_s])
                
                # Update value function
                self.value_function[s] = new_value
                delta = max(delta, abs(v - new_value))
    
    def _policy_improvement(self):
        """Improve the policy based on the current value function."""
        policy_stable = True
        for s in range(self.env.n_states):
            old_action = self.policy[s]
            
            # Calculate action values
            action_values = np.zeros(len(self.env.action_space))
            for a in self.env.action_space:
                # Get reward
                reward = -self.env.h(s) if a == 0 else -self.env.K
                
                # Calculate expected value for this action
                expected_value = 0
                for next_s in range(self.env.n_states):
                    prob = self.env.P[a, s, next_s]
                    expected_value += prob * (reward + self.gamma * self.value_function[next_s])
                
                action_values[a] = expected_value
            
            # Choose best action
            best_action = np.argmax(action_values)
            
            # Update policy
            self.policy[s] = best_action
            
            # Check if policy changed
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    
    def compute_optimal_value(self, start_state: int) -> float:
        """
        Compute the expected value of following the current policy from start_state.
        This uses a Monte Carlo simulation similar to the Q-Learning implementation.
        """
        state = start_state
        total_reward = 0
        discount = 1.0
        
        # Save the current environment state to reset later
        env_state = self.env.state
        self.env.reset()
        
        for _ in range(self.env.max_steps):
            action = self.policy[state]
            next_state, reward, done, _ = self.env.step(action)
            total_reward += discount * reward
            discount *= self.gamma
            state = next_state
            if done:
                break
        
        # Reset environment to original state
        self.env.reset()
        for _ in range(env_state):
            self.env.step(0)  # Assuming this moves the state forward
            
        return total_reward
    
    def train(self, n_episodes: int = 1000):
        """Train the agent using policy iteration."""
        iteration = 0
        policy_stable = False
        
        # For tracking best policy
        best_value = float('-inf')
        best_policy = None
        patience_counter = 0
        
        while not policy_stable and iteration < n_episodes:
            # Policy evaluation
            self._policy_evaluation()
            
            # Policy improvement
            intermediate = self._policy_improvement()
            if intermediate:
                patience_counter +=1
            else:
                patience_counter = 0
            if patience_counter > 100:
                policy_stable = True
            # self._policy_improvement()
            
            # Compute actual discounted reward by simulating the policy
            discounted_reward = self.compute_optimal_value(0)
            self.episode_rewards.append(discounted_reward)
            
            # Update best policy if needed
            if discounted_reward > best_value:
                best_value = discounted_reward
                best_policy = self.policy.copy()
            
            # Calculate regret (similar to Q-learning implementation)
            if len(self.episode_rewards) > 1:
                prev_best = max(self.episode_rewards[:-1])
                regret = max(0, prev_best - discounted_reward)
            else:
                regret = 0
                
            self.regrets.append(regret)
            self.cumulative_regrets.append(sum(self.regrets))
            self.cumulative_rewards.append(sum(self.episode_rewards))
            
            iteration += 1
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Value: {discounted_reward:.2f}, Policy stable: {policy_stable}")
        
        # Set policy to best found
        if best_policy is not None:
            self.policy = best_policy
        
        print(f"Policy iteration completed after {iteration} iterations")
        print(f"Final Policy:\n{self.policy}")
        print(f"Final Value Function:\n{self.value_function}")
            
    def moving_average(self, data: list, window: int = 10) -> np.ndarray:
        """Compute moving average of data with given window size"""
        if len(data) < window:
            return np.array(data)
        weights = np.ones(window) / window
        return np.convolve(data, weights, mode='valid')

    def plot_metrics(self):
        """Plot training metrics with moving averages"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot episode rewards
        ax1.plot(self.episode_rewards, alpha=0.6, label='Raw')
        if len(self.episode_rewards) > 10:  # Only plot MA if enough data
            ma_rewards = self.moving_average(self.episode_rewards)
            ax1.plot(range(9, len(self.episode_rewards)), 
                    ma_rewards, 'r', label='50-ep Moving Avg')
        ax1.set_title('Discounted Rewards per Iteration')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Discounted Reward')
        ax1.legend()
        
        # Plot cumulative rewards
        ax2.plot(self.cumulative_rewards)
        ax2.set_title('Cumulative Rewards')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Cumulative Reward')
        
        # Plot regrets
        ax3.plot(self.regrets, alpha=0.6, label='Raw')
        if len(self.regrets) > 10:  # Only plot MA if enough data
            ma_regrets = self.moving_average(self.regrets)
            ax3.plot(range(9, len(self.regrets)), 
                    ma_regrets, 'r', label='50-ep Moving Avg')
        ax3.set_title('Instantaneous Regret')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Regret')
        ax3.legend()
        
        # Plot cumulative regrets
        ax4.plot(self.cumulative_regrets)
        ax4.set_title('Cumulative Regret')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Cumulative Regret')
        
        # Plot the policy
        fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Policy visualization
        states = np.arange(self.env.n_states)
        ax5.bar(states, self.policy, color=['blue' if a == 0 else 'red' for a in self.policy])
        ax5.set_title('Optimal Policy')
        ax5.set_xlabel('Machine State')
        ax5.set_ylabel('Action')
        ax5.set_yticks([0, 1])
        ax5.set_yticklabels(['Continue', 'Replace'])
        
        # Value function visualization
        ax6.plot(states, self.value_function, 'go-')
        ax6.set_title('Value Function')
        ax6.set_xlabel('Machine State')
        ax6.set_ylabel('State Value')
        
        plt.tight_layout()
        plt.show()




env = MachineReplacementEnv()

# Create and train agent

agent = PolicyIterationAgent(env)
agent.train(1000)
agent.plot_metrics()