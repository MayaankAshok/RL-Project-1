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
        self.policy_history = []
        self.value_function_history = []

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
        self.value_function_history.append(self.value_function.copy())
    
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
        
        self.policy_history.append(self.policy.copy())
        return policy_stable
    
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
            if not self._policy_improvement():
                policy_stable = False
            else:
                policy_stable = True
            iteration += 1
            

        
        print(f"Policy iteration completed after {iteration} iterations")
        print(f"Final Policy:\n{self.policy}")
        print(f"Final Value Function:\n{self.value_function}")
            

    def plot_metrics(self):
        """Plot the policy and value function for every iteration."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Policy visualization
        for i, policy in enumerate(self.policy_history):
            ax1.plot(policy + i * 0.1, 'o-', label=f'Iter {i}')
        ax1.set_title('Policy Evolution')
        ax1.set_xlabel('Machine State')
        ax1.set_ylabel('Action')
        ax1.legend()
        
        # Value function visualization
        for i, value_function in enumerate(self.value_function_history):
            ax2.plot(value_function)
        ax2.set_title('Value Function Evolution')
        ax2.set_xlabel('Machine State')
        ax2.set_ylabel('State Value')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_policy_and_value_function(self):
        """Plot the policy and value function at each iteration"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Policy visualization
        states = np.arange(self.env.n_states)
        ax1.bar(states, self.policy, color=['blue' if a == 0 else 'red' for a in self.policy])
        ax1.set_title('Optimal Policy')
        ax1.set_xlabel('Machine State')
        ax1.set_ylabel('Action')
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['Continue', 'Replace'])
        
        # Value function visualization
        ax2.plot(states, self.value_function, 'go-')
        ax2.set_title('Value Function')
        ax2.set_xlabel('Machine State')
        ax2.set_ylabel('State Value')
        
        plt.tight_layout()
        plt.show()




env = MachineReplacementEnv()

# Create and train agent

agent = PolicyIterationAgent(env)
agent.train()
agent.plot_metrics()
agent.plot_policy_and_value_function()