#!/usr/bin/env python3
"""
This file contains skeleton code for implementing two reinforcement learning algorithms:
1. Active ADP (Model-Based RL) - Learn environment model and plan with value iteration
2. Q-Learning (Model-Free RL) - Learn Q-values directly from experience

- Complete the TODO sections in ActiveADPLearner class:
- Complete the TODO sections in QLearner class:

Both algorithms use optimistic exploration: f(u,n) = R_plus if n < Ne else u
"""

import argparse, random, os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
from utils import record_video, plot_learning_curves, live_animation, evaluate, make_env

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class ActiveADPLearner:
    """
    Model-based RL (Active ADP) with planning and optimistic action selection:
      f(u,n) = R_plus if n < Ne else u.
    Learn P(s'|s,a) and R(s,a) from experience; plan with value iteration;
    improve policy by choosing argmax_a f(Q_model(s,a), Nsa[s,a]).
    """

    def __init__(self, nS, nA, gamma=0.99, R_plus=5.0, Ne=10,
                 plan_iters=100, plan_tol=1e-8, tie_break_random=True):
        self.nS, self.nA = int(nS), int(nA)
        self.gamma = float(gamma)
        self.R_plus = float(R_plus)
        self.Ne = int(Ne)
        self.plan_iters = int(plan_iters)
        self.plan_tol = float(plan_tol)
        self.tie_break_random = bool(tie_break_random)

        # Learned model
        self.Nsa = np.zeros((self.nS, self.nA), dtype=np.int64)      
        self.Nsas = np.zeros((self.nS, self.nA, self.nS), dtype=np.int64) 
        self.Rsum = np.zeros((self.nS, self.nA), dtype=np.float64)    

        # Estimates
        self.P = np.zeros((self.nS, self.nA, self.nS), dtype=np.float64)   
        self.R = np.zeros((self.nS, self.nA), dtype=np.float64)            
        self.U = np.zeros(self.nS, dtype=np.float64)                       

        # Previous transition
        self.s = None
        self.a = None

    # --------- Public API ---------

    def start_episode(self, s0):
        """Call at the beginning of each episode with env.reset() result s0."""
        self.s = int(s0)
        
        self._plan_value_iteration()         
        a0 = self._select_action(self.s)
        self.a = a0
        return a0

    def step(self, percept):
        """
        One Active ADP step.
        percept = (s_prime, r_prime, done)
        Returns next action a' (or None if done).
        """
        # TODO: Implement Active ADP model learning
        pass


    def _plan_value_iteration(self):
        """
        Value iteration with the learned model:
          U(s) <- max_a [ R(s,a) + gamma * sum_{t} P(t|s,a) * U(t) ].
        Runs a small number of iterations online; converges as experience grows.
        """
        # TODO: Implement value iteration planning
        pass

    def _select_action(self, s):
        """
        Optimistic policy improvement:
          a* = argmax_a f(Q_model(s,a), Nsa[s,a]),
        where Q_model(s,a) = R(s,a) + gamma * P(s,a,:) dot U,
              f(u,n) = R_plus if n < Ne else u.
        """
        # TODO: Implement optimistic action selection
        pass



def run_active_adp(env, steps=1000, eval_every=10, deterministic=False, success_rate=1, **agent_kwargs):
    """Run Active ADP with optimistic exploration."""
    nS, nA = env.observation_space.n, env.action_space.n
    agent = ActiveADPLearner(nS=nS, nA=nA, **agent_kwargs)
    
    # Track learning progress
    returns_history = []
    step_counts = []
    
    step_count = 0
    episode_count = 0
    
    while step_count < steps:
        # Start new episode
        obs, _ = env.reset(seed=episode_count)
        s = obs
        a = agent.start_episode(s)
        done = False
        episode_return = 0
        
        while not done and step_count < steps:
            # Take action and observe transition
            obs2, r, terminated, truncated, _ = env.step(a)
            s2 = obs2
            done = terminated or truncated
            episode_return += r
            step_count += 1
            
            # Update agent with the transition
            a = agent.step((s2, r, done))
            
            # Evaluate every eval_every steps
            if step_count % eval_every == 0:
                # Evaluate using the learned model
                def model_based_action(state):
                    # Use the learned model to select actions
                    q_row = agent.R[state, :] + agent.gamma * agent.P[state, :, :].dot(agent.U)
                    n_row = agent.Nsa[state, :]
                    adjusted = np.where(n_row < agent.Ne, agent.R_plus, q_row)
                    return int(np.argmax(adjusted))
                
                avg_return = evaluate(make_env, model_based_action, episodes=50, deterministic=deterministic, success_rate=success_rate)
                returns_history.append(avg_return)
                step_counts.append(step_count)
        
        episode_count += 1
    
    # Return the model-based policy
    def model_policy(state):
        q_row = agent.R[state, :] + agent.gamma * agent.P[state, :, :].dot(agent.U)
        n_row = agent.Nsa[state, :]
        adjusted = np.where(n_row < agent.Ne, agent.R_plus, q_row)
        return int(np.argmax(adjusted))
    
    return model_policy, agent, (step_counts, returns_history)



class QLearner:
    """
    Q-learning with optimistic exploration:
      f(u,n) = R_plus if n < Ne else u   (used only for action selection)
    """

    def __init__(self, nS, nA, gamma=0.99, alpha=0.05, R_plus=5.0, Ne=10, tie_break_random=True):
        self.nS, self.nA   = nS, nA
        self.gamma         = float(gamma)
        self.alpha         = float(alpha)     
        self.R_plus        = float(R_plus)    
        self.Ne            = int(Ne)          
        self.tie_break_random = bool(tie_break_random)

        
        self.Q   = np.zeros((nS, nA), dtype=np.float64)
        self.Nsa = np.zeros((nS, nA), dtype=np.int64)

        
        self.s = None
        self.a = None


    def start_episode(self, s0):
        """Call at the beginning of each episode with env.reset() result s0."""
        self.s = int(s0)
        self.a = self._select_action(self.s)
        return self.a

    def step(self, percept):
        """
        One Q-learning step.
        percept = (s_prime, r_prime, done)
        Returns next action a' (or None if done).
        """
        # TODO: Implement Q-learning step
        pass


    def _select_action(self, s):
        """
        f(u,n) = R_plus if n < Ne else u  (optimistic exploration).
        Break ties randomly if tie_break_random=True.
        """
        # TODO: Implement optimistic action selection for Q-learning
        pass


def run_ql(env, steps=1000, eval_every=10, deterministic=False, success_rate=1, **agent_kwargs):
    """Run Q-learning with optimistic exploration."""
    nS, nA = env.observation_space.n, env.action_space.n
    agent = QLearner(nS=nS, nA=nA, **agent_kwargs)
    
    # Track learning progress
    returns_history = []
    step_counts = []
    
    step_count = 0
    episode_count = 0
    
    while step_count < steps:
        # Start new episode
        obs, _ = env.reset(seed=episode_count)
        s = obs
        a = agent.start_episode(s)
        done = False
        episode_return = 0
        
        while not done and step_count < steps:
            # Take action and observe transition
            obs2, r, terminated, truncated, _ = env.step(a)
            s2 = obs2
            done = terminated or truncated
            episode_return += r
            step_count += 1
            
            # Update agent with the transition
            a = agent.step((s2, r, done))
            
            # Evaluate every eval_every steps
            if step_count % eval_every == 0:
                avg_return = evaluate(make_env, lambda s: int(np.argmax(agent.Q[s])), episodes=50, deterministic=deterministic, success_rate=success_rate)
                returns_history.append(avg_return)
                step_counts.append(step_count)
        
        episode_count += 1
    
    def greedy(s): 
        return int(np.argmax(agent.Q[s]))
    return greedy, agent, (step_counts, returns_history)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ql","active_adp","plot"], required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--eval_episodes", type=int, default=50)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--success_rate", type=float, default=1, help="Set success rate (0.0=no success, 1.0=always successful)")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--live", action="store_true")
    args=ap.parse_args()

    deterministic = args.deterministic
    
    if args.algo=="plot":
        plot_learning_curves(steps=args.steps, deterministic=deterministic, success_rate=args.success_rate)
    else:
        if args.algo=="active_adp":
            greedy,_,_=run_active_adp(make_env(deterministic=deterministic, success_rate=args.success_rate), steps=args.steps, deterministic=deterministic, success_rate=args.success_rate)
        elif args.algo=="ql":
            greedy,_,_=run_ql(make_env(deterministic=deterministic, success_rate=args.success_rate), steps=args.steps, deterministic=deterministic, success_rate=args.success_rate)

        avg=evaluate(make_env, greedy, episodes=args.eval_episodes, deterministic=deterministic, success_rate=args.success_rate)
        print(f"[{args.algo.upper()}] Avg return over {args.eval_episodes} episodes: {avg:.3f}")
        if args.record: record_video(greedy, deterministic=deterministic, episodes=1, algo_name=args.algo)
        if args.live: live_animation(greedy, deterministic=deterministic, steps=150)

if __name__=="__main__":
    main()