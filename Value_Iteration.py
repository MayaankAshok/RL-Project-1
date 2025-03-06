import numpy as np
import matplotlib.pyplot as plt
from Environment import MachineReplacementEnv

class ValueIterationAgent:
    def __init__(self, env: MachineReplacementEnv, discount_factor: float = 0.95, theta: float = 1e-6):
        self.env = env
        self.gamma = discount_factor
        self.theta = theta  # end conition for convergence threshol
        self.value_function = np.zeros(env.n_states)
        self.policy = np.zeros(env.n_states, dtype=int) # initialization
        self.value_history = []
        self.instantaneous_rewards = []
        self.cumulative_rewards = []
        self.instantaneous_regret = []
        self.cumulative_regret = []
    
    def one_step_lookahead(self, state: int) -> np.ndarray:
        action_values = np.zeros(len(self.env.action_space))
        for action in self.env.action_space:
            if action == 0:  # continue without replacement
                prob_next = [1 - self.env.p, self.env.p]  # Probabilities for no wear and wear
                next_states = [state, min(state + 1, self.env.n_states - 1)] # next states calcualation from given eq
                rewards = [-self.env.h(state), -self.env.h(state)] # rewards for no wear
            else:  # replacing the machine
                prob_next = [1] # probability of next state made as 1
                next_states = [0]
                rewards = [-self.env.K]
            
            action_values[action] = sum(prob * (reward + self.gamma * self.value_function[next_state])for prob, next_state, reward in zip(prob_next, next_states, rewards)) # Bellman Equation
        return action_values
    
    def value_iteration(self, max_iterations: int = 5000):
        cumulative_reward = 0
        cumulative_regret = 0
        
        for _ in range(max_iterations):
            delta = 0
            new_value_function = np.zeros_like(self.value_function)
            total_reward = 0
            
            for state in range(self.env.n_states):
                action_values = self.one_step_lookahead(state) # Calculate the action values
                best_action_value = max(action_values) # select the best action
                new_value_function[state] = best_action_value # update the value function
                delta = max(delta, abs(best_action_value - self.value_function[state])) # calculate the change in value function
                total_reward += best_action_value
                
            self.value_function = new_value_function
            self.value_history.append(self.value_function.copy())
            self.instantaneous_rewards.append(total_reward)
            cumulative_reward += total_reward
            self.cumulative_rewards.append(cumulative_reward) # saving vales for graphs
            
            if len(self.value_history) > 1:
                regret = np.linalg.norm(self.value_history[-1] - self.value_history[-2]) # calculate the regret
                self.instantaneous_regret.append(regret)
                cumulative_regret += regret
                self.cumulative_regret.append(cumulative_regret)
            else:
                self.instantaneous_regret.append(0)
                self.cumulative_regret.append(0)
            
            if delta < self.theta:  # Convergence condition
                break
        
        # Extract optimal policy
        for state in range(self.env.n_states):
            self.policy[state] = np.argmax(self.one_step_lookahead(state)) # select the best action
    
    def plot_metrics(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        for i, values in enumerate(self.value_history[::len(self.value_history)//10]):
            axs[0, 0].plot(values, label=f'Iter {i * (len(self.value_history)//10)}')
        axs[0, 0].set_xlabel("State")
        axs[0, 0].set_ylabel("Value")
        axs[0, 0].set_title("Value Function Convergence")
        axs[0, 0].legend()
        
        axs[0, 1].plot(self.instantaneous_rewards, label="Instantaneous Reward", color='blue')
        axs[0, 1].set_xlabel("Iteration")
        axs[0, 1].set_ylabel("Reward")
        axs[0, 1].set_title("Instantaneous Rewards Over Iterations")
        axs[0, 1].legend()
        
        axs[1, 0].plot(self.cumulative_rewards, label="Cumulative Reward", color='green')
        axs[1, 0].set_xlabel("Iteration")
        axs[1, 0].set_ylabel("Cumulative Reward")
        axs[1, 0].set_title("Cumulative Rewards Over Iterations")
        axs[1, 0].legend()
        
        axs[1, 1].plot(self.instantaneous_regret, label="Instantaneous Regret", color='red')
        axs[1, 1].plot(self.cumulative_regret, label="Cumulative Regret", color='purple')
        axs[1, 1].set_xlabel("Iteration")
        axs[1, 1].set_ylabel("Regret")
        axs[1, 1].set_title("Instantaneous and Cumulative Regret Over Iterations")
        axs[1, 1].legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    env = MachineReplacementEnv()
    agent = ValueIterationAgent(env)
    agent.value_iteration()
    print("Optimal Policy:", agent.policy)
    agent.plot_metrics()
