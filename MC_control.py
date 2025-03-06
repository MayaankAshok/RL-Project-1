import numpy as np
import matplotlib.pyplot as plt
from Environment import MachineReplacementEnv

# np.random.seed(42)  # Set the random seed for reproducibility

class MCControl:
    def __init__(self, env, gamma=0.9, epsilon=0.30, episodes=5000):
        self.env = env
        self.gamma = gamma        # Discount factor
        self.epsilon = epsilon    # Exploration probability for epsilon-greedy policy.
        self.episodes = episodes  # Number of episodes to run.
        self.Q = np.zeros((env.n_states, len(env.action_space)))   # Q-table: stores the estimated return for each state-action pair.
        self.returns = {s: {a: [] for a in env.action_space} for s in env.observation_space}   # Stores the returns for each (state, action) pair
        self.policy = np.zeros(env.n_states, dtype=int)   # each state, the action with the highest Q-value.

        # Metrics for visualization
        self.rewards_per_episode = []   # Total (discounted) reward per episode.
        self.regret_per_episode = []    # Instantaneous regret (difference between the optimal and actual reward).
        self.cumulative_regret = []     # Running sum of instantaneous regrets.
        
    def epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.action_space)    # Explore
        else:
            return np.argmax(self.Q[state])             # Exploit

    def compute_optimal_value(self, tolerance=1e-3, max_iter=1000):
        """
        Compute the optimal value function V* using value iteration.
        The Bellman update for each state s is:
        
        Q_continue = -h(s) + gamma * [ (1-p)*V(s) + p*V(min(s+1, n-1)) ]
        Q_replace  = -K + gamma * V(0)
        V*(s) = max(Q_continue, Q_replace)
        """
        n = self.env.n_states
        p = self.env.p
        K = self.env.K
        gamma = self.gamma
        V = np.zeros(n)
        
        for _ in range(max_iter):
            V_new = np.zeros(n)
            for s in range(n):
                cost_operate = -self.env.h(s)
                # When operating, next state is s (with probability 1-p) or min(s+1, n-1) (with probability p)
                next_val = (1 - p) * V[s] + p * V[min(s + 1, n - 1)]
                Q_continue = cost_operate + gamma * next_val
                # When replacing, cost is -K and machine resets to state 0.
                Q_replace = -K + gamma * V[0]
                V_new[s] = max(Q_continue, Q_replace)
            if np.max(np.abs(V_new - V)) < tolerance:
                V = V_new
                break
            V = V_new
        return V

    def train(self):
        cumulative_regret_val = 0
        
        for episode in range(1, self.episodes + 1):
            state = self.env.reset()
            episode_data = []
            total_reward = 0      # Sum of rewards (non-discounted) for reference
            discount = 1.0
            discounted_reward = 0
            
            # Run the episode until 'done' is True
            while True:
                action = self.epsilon_greedy(state)
                next_state, reward, done, info = self.env.step(action)
                episode_data.append((state, action, reward))
                total_reward += reward
                discounted_reward += discount * reward
                discount *= self.gamma
                state = next_state
                if done:
                    break
            
            # Store the discounted reward for this episode.
            self.rewards_per_episode.append(discounted_reward)
            
            # Compute instantaneous regret:
            # For an episode starting at state 0, the optimal expected reward is V*(0)
            optimal_values = self.compute_optimal_value()
            optimal_reward = optimal_values[0]  # since env.reset() always returns 0
            instantaneous_regret = optimal_reward - discounted_reward
            cumulative_regret_val += instantaneous_regret
            
            self.regret_per_episode.append(instantaneous_regret)
            self.cumulative_regret.append(cumulative_regret_val)
            
            # Monte Carlo update: first-visit MC control
            G = 0
            visited = set()
            for state, action, reward in reversed(episode_data):
                G = reward + self.gamma * G
                if (state, action) not in visited:
                    self.returns[state][action].append(G)
                    self.Q[state, action] = np.mean(self.returns[state][action])
                    self.policy[state] = np.argmax(self.Q[state])
                    visited.add((state, action))
            
            if episode % 100 == 0:
                print(f"Episode {episode}/{self.episodes}")

    def plot_results(self):
        plt.figure(figsize=(10, 8))
        
        # Discounted Rewards per Episode
        plt.subplot(2, 2, 1)
        plt.plot(self.rewards_per_episode, alpha=0.5, label='Raw')
        plt.plot(np.convolve(self.rewards_per_episode, np.ones(50)/50, mode='valid'),
                 label='50-ep Moving Avg', color='red')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Discounted Rewards per Episode')
        plt.legend()
        
        # Cumulative Rewards
        plt.subplot(2, 2, 2)
        plt.plot(np.cumsum(self.rewards_per_episode))
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards')
        
        # Instantaneous Regret
        plt.subplot(2, 2, 3)
        plt.plot(self.regret_per_episode, alpha=0.5, label='Raw')
        plt.plot(np.convolve(self.regret_per_episode, np.ones(50)/50, mode='valid'),
                 label='50-ep Moving Avg', color='red')
        plt.xlabel('Episode')
        plt.ylabel('Regret')
        plt.title('Instantaneous Regret')
        plt.legend()
        
        # Cumulative Regret
        plt.subplot(2, 2, 4)
        plt.plot(self.cumulative_regret)
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Regret')
        plt.title('Cumulative Regret')
        
        plt.tight_layout()
        plt.show()

    def plot_q_values(self):
        states = np.arange(self.env.n_states)
        q_operate = self.Q[:, 0]  # Assuming "operate" is action 0
        q_replace = self.Q[:, 1]  # Assuming "replace" is action 1

        plt.figure(figsize=(10, 5))
        plt.plot(states, q_operate, label="Operate", marker='o')
        plt.plot(states, q_replace, label="Replace", marker='s')

        plt.xlabel("State (Deterioration Level)")
        plt.ylabel("Q-Value")
        plt.title("Q-values for 'Operate' vs 'Replace'")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    env = MachineReplacementEnv()
    mc_control = MCControl(env)
    mc_control.train()
    mc_control.plot_results()
    mc_control.plot_q_values()













