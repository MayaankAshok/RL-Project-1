# Topics in RL - Project 1

February 2025

**Common Instructions**

For each project, you should implement any 4 of the following 5 algorithms using only python to find the optimal policy:

1. Value iteration  
2. Policy iteration  
3. MC control (on and/or off)  
4. TD control using SARSA(λ)  
5. Q learning  

Do not use any RL packages or ready made commands/codes for any of the above algorithms. We want to see a code from scratch. For the first and second algorithm, assume transition probabilities and rewards are known and then show that the algorithms converge to the optimal value function/policy. For rest of the algorithms, use the transition probabilities to only generate the episodes online. For algorithms 3-5, plot the total discounted reward per episode and cumulative rewards over episodes. Also plot the instantaneous and cumulative regrets for the RL algorithms defined as follows.  

**Instantaneous episodic regret**: It is the difference between the total discounted reward earned by your RL algorithm in the current episode and the expected cumulative discounted reward earned by the optimal policy in an episode (essentially \(V^\pi(s)\) if the episode started in state \(s\)). It measures how much reward you are losing choosing a sub-optimal policy.  

**Cumulative regret**: At the current iteration/episode, this is running/cumulative sum of all the previous instantaneous episodic regrets till now.  

**Value and Policy iteration for the finite time horizon**: Recall from the class that VI and PI were used for infinite horizon discounted problem settings. In fact, in such settings deterministic, stationary and Markovian policies are optimal and so in policy iteration we always chose deterministic and stationary policy. We also know that for a finite horizon problem, the optimal policy is non-stationary. In that case, how will you adapt that equation to solve the finite time horizon problem? Give it a thought! The following paragraph with Q learning for finite horizon will act as a hint.  

**Q-learning for finite horizon**: Q-learning in the standard form assumes deterministic and stationary policy for an infinite horizon MDP. In many of the projects, the problem only makes sense for finite horizons. You can adapt Q-learning to finite horizon as shown in this link. Instead of assuming stationary policy and using \(Q(s, a)\), use the notation \(Q_t(s, a)\), and make the update to it in a similar fashion. See the following algorithm from the above paper.  

---

## Project 3: A machine replacement model (Team 8)  

Consider a manufacturing process, where the machine used for manufacturing deteriorates over time. Let \(\mathcal{S}=\{0,1,\ldots,n\}\) represent the condition of the machine. The higher the value of \(s\), the worse the condition of the equipment.  

A decision maker observes the state of the machine and has two options: continue operating the machine or replace it with a new and identical piece of equipment. Operating the machine in state \(s\) costs \(h(s)\), where \(h(\cdot)\) is a weakly increasing function; replacing the machine costs a constant amount \(K\).  

When the machine is operated, its state deteriorates according to:  

\[
S_{t+1}=\min(S_{t}+W_{t},n)
\]  

where \(\{W_{t}\}_{t\geq 1}\) is an i.i.d. discrete process.  

You can assume \(W_{t}\sim\text{Bernoulli}(p)\) for some \(p\in(0,1)\). Also, take \(K\in(h(s_{i}),h(s_{i+1}))\) for some \(s_{i}\in S\). And \(h(x)=x^{2}\).  

**Reference**: Aditya Mahajan Notes on Machine Replacement.  