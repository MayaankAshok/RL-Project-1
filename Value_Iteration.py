import numpy as np
import matplotlib.pyplot as plt
from Environment import MachineReplacementEnv
import time

class ValueIterationAgent:
    def __init__(self, env: MachineReplacementEnv, discount_factor: float = 0.95, theta: float = 1e-6):
        self.env = env
        self.gamma = discount_factor
        self.theta = theta  # Convergence threshold

        self.value_function = np.zeros(env.n_states)
        self.policy = np.zeros(env.n_states, dtype=int) # initialization
            
        # to Track metrics
        self.value_history = []
        self.bellman_errors = []
        self.execution_times = []
    
    def one_step_lookahead(self, state: int) -> np.ndarray:
        action_values = np.zeros(len(self.env.action_space))
        
        for action in self.env.action_space:
            if action == 0:  # Continue
                prob_next = [1 - self.env.p, self.env.p]  # Probabilities for no wear and wear
                next_states = [state, min(state + 1, self.env.n_states - 1)] # Next states for no wear and wear
                rewards = [-self.env.h(state), -self.env.h(state)] # Rewards for no wear and wear
            else:  # Replace
                prob_next = [1] # Probability 1 for replacement
                next_states = [0] # Next state is 0 after replacement
                rewards = [-self.env.K]
            
            action_values[action] = sum(prob * (reward + self.gamma * self.value_function[next_state])for prob, next_state, reward in zip(prob_next, next_states, rewards)) # Bellman Equation update step
        
        return action_values
    
    def value_iteration(self, max_iterations: int = 1000):

        for _ in range(max_iterations):
            start_time = time.time()
            delta = 0
            new_value_function = np.zeros_like(self.value_function)
            
            for state in range(self.env.n_states):
                action_values = self.one_step_lookahead(state) # Compute the value function for all actions
                new_value_function[state] = max(action_values) # select the best action value and udate the value function
                delta = max(delta, abs(new_value_function[state] - self.value_function[state])) # calculate the maximum difference between the new and old value function
                
            self.value_function = new_value_function
            self.value_history.append(self.value_function.copy())
            self.bellman_errors.append(delta)
            self.execution_times.append(time.time() - start_time)
            
            if delta < self.theta:  # Convergence condition
                break

        for state in range(self.env.n_states):
            self.policy[state] = np.argmax(self.one_step_lookahead(state)) # Find the best action for each state
    
    def plot_metrics(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].plot(self.bellman_errors)
        axs[0, 0].set_title("Bellman Error vs Iterations")
        axs[0, 0].set_xlabel("Iteration")
        axs[0, 0].set_ylabel("Max Value Difference")

        for i, values in enumerate(self.value_history[::len(self.value_history)//10]):axs[0,1].plot(values, label=f'Iter {i * (len(self.value_history)//10)}') # Plotting every 10th value function
        axs[0,1].legend()
        axs[0,1].set_title("Value Function Convergence")
        axs[0,1].set_xlabel("State")
        axs[0,1].set_ylabel("Value")

        value_differences = [np.sum(np.abs(self.value_history[i] - self.value_history[i-1])) for i in range(1, len(self.value_history))] # Calculate the sum of value differences between consecutive iterations
        axs[1, 0].plot(value_differences)
        axs[1, 0].set_title("Cumulative Value Difference per Iteration")
        axs[1, 0].set_xlabel("Iteration")
        axs[1, 0].set_ylabel("Sum of Value Changes")

        axs[1, 1].plot(self.policy, marker='o')
        axs[1, 1].set_title("Optimal Policy")
        axs[1, 1].set_xlabel("State")
        axs[1, 1].set_ylabel("Action (0 = Continue, 1 = Replace)")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    env = MachineReplacementEnv()
    agent = ValueIterationAgent(env)
    agent.value_iteration()
    print("Optimal Policy:", agent.policy)
    agent.plot_metrics()
