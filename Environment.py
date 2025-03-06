import numpy as np
from typing import Tuple, Optional
from scipy.sparse import csr_matrix

class MachineReplacementEnv:
    def __init__(
        self,
        n_states: int = 6,
        replacement_factor: float = 0.6,
        p: float = 0.3,
        max_steps: int = 100
    ):
        """
        Initialize Machine Replacement Environment
        
        Args:
            n_states: Number of states (0 to n-1)
            replacement_cost: Cost K for replacing machine
            p: Probability parameter for Bernoulli distribution
            max_steps: Maximum steps per episode
        """
        self.n_states = n_states
        self.K = self.h(n_states - 1) * replacement_factor
        self.p = p
        self.max_steps = max_steps
        
        self.action_space = [0, 1]  # 0: continue, 1: replace
        self.observation_space = list(range(n_states))
        
        self.create_TPM()
        self.reset()

    def create_TPM(self) -> np.ndarray:
        """Create Transition Probability Matrix for the environment"""
        self.P = np.zeros((2, self.n_states, self.n_states))  # P[action, s, s']
        self.q = 1 - self.p
        for i in range(self.n_states):
            # If action is continue (0), stays with prob q, degrades with prob p

            self.P[0, i, i] = self.q
            if i < self.n_states - 1:
                self.P[0, i, i + 1] = self.p
            else:
                self.P[0, i, i] = 1  # If already at worst state, stays there
            
            # If action is replace (1), always resets to state 0
            self.P[1, i, 0] = 1
    
    def h(self, s: int) -> float:
        """Operating cost function h(s) = s^2"""
        return float(s ** 2)
    
    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Take action in environment
        
        Args:
            action: 0 for continue, 1 for replace
            
        Returns:
            (next_state, reward, done, info)
        """
        assert action in self.action_space
        
        # Get reward based on action
        reward = -self.h(self.state) if action == 0 else -self.K
        
        # Update state
        if action == 0:  # Continue
            w = np.random.binomial(1, self.p)  # Bernoulli trial
            self.state = min(self.state + w, self.n_states - 1)
        else:  # Replace
            self.state = 0
            
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info = {
            "operating_cost": self.h(self.state),
            "step": self.current_step
        }
        
        return self.state, reward, done, info
    
    def reset(self) -> int:
        """Reset environment to initial state"""
        self.state = 0  # Start with new machine
        self.current_step = 0
        return self.state
    
    def render(self):
        """Simple console rendering"""
        print(f"Current machine state: {self.state}")
