#!/usr/bin/env python3
"""
This file contains skeleton code for implementing two reinforcement learning algorithms:
1. Active ADP (Model-Based RL) - Learn environment model and plan with value iteration
2. Q-Learning (Model-Free RL) - Learn Q-values directly from experience

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
    Model-based RL (Active ADP) with planning and optimistic action selection.
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

        # Model estimates
        self.P = np.zeros((self.nS, self.nA, self.nS), dtype=np.float64)
        self.R = np.zeros((self.nS, self.nA), dtype=np.float64)
        self.U = np.zeros(self.nS, dtype=np.float64)

        self.s = None
        self.a = None

    # --------- Public API ---------

    def start_episode(self, s0):
        self.s = int(s0)
        self._plan_value_iteration()
        a0 = self._select_action(self.s)
        self.a = a0
        return a0

    def step(self, percept):
        """
        Active ADP update + planning + action selection.
        """
        s_prime, r_prime, done = percept
        s_prime = int(s_prime)

        s = self.s
        a = self.a

        # --- Model learning ---
        self.Nsa[s, a] += 1
        self.Nsas[s, a, s_prime] += 1
        self.Rsum[s, a] += r_prime

        n_sa = self.Nsa[s, a]
        if n_sa > 0:
            self.P[s, a, :] = self.Nsas[s, a, :] / n_sa
            self.R[s, a] = self.Rsum[s, a] / n_sa

        # --- Planning ---
        self._plan_value_iteration()

        if done:
            self.s = None
            self.a = None
            return None

        # --- Next action ---
        self.s = s_prime
        a_prime = self._select_action(self.s)
        self.a = a_prime
        return a_prime

    def _plan_value_iteration(self):
        """
        Value iteration on learned model.
        """
        for _ in range(self.plan_iters):
            U_old = self.U.copy()

            # Q(s,a) = R(s,a) + γ * Σ_t P(t|s,a)*U_old[t]
            Q = self.R + self.gamma * np.tensordot(self.P, U_old, axes=([2], [0]))
            self.U = np.max(Q, axis=1)

            if np.max(np.abs(self.U - U_old)) < self.plan_tol:
                break

    def _select_action(self, s):
        """
        a* = argmax_a f(Q_model(s,a), Nsa[s,a])
        f(u,n) = R_plus if n < Ne else u
        """
        s = int(s)

        q_row = self.R[s, :] + self.gamma * self.P[s, :, :].dot(self.U)
        n_row = self.Nsa[s, :]
        adjusted = np.where(n_row < self.Ne, self.R_plus, q_row)

        if self.tie_break_random:
            max_val = np.max(adjusted)
            candidates = np.flatnonzero(np.isclose(adjusted, max_val))
            return int(np.random.choice(candidates))
        return int(np.argmax(adjusted))



def run_active_adp(env, steps=1000, eval_every=10, deterministic=False, success_rate=1, **agent_kwargs):
    nS, nA = env.observation_space.n, env.action_space.n
    agent = ActiveADPLearner(nS=nS, nA=nA, **agent_kwargs)

    returns_history = []
    step_counts = []
    step_count = 0
    episode_count = 0

    while step_count < steps:
        obs, _ = env.reset(seed=episode_count)
        s = obs
        a = agent.start_episode(s)
        done = False
        episode_return = 0

        while not done and step_count < steps:
            obs2, r, terminated, truncated, _ = env.step(a)
            s2 = obs2
            done = terminated or truncated
            episode_return += r
            step_count += 1

            a = agent.step((s2, r, done))

            if step_count % eval_every == 0:
                def model_based_action(state):
                    q_row = agent.R[state, :] + agent.gamma * agent.P[state, :, :].dot(agent.U)
                    n_row = agent.Nsa[state, :]
                    adjusted = np.where(n_row < agent.Ne, agent.R_plus, q_row)
                    return int(np.argmax(adjusted))

                avg_return = evaluate(make_env, model_based_action,
                                      episodes=50, deterministic=deterministic,
                                      success_rate=success_rate)
                returns_history.append(avg_return)
                step_counts.append(step_count)

        episode_count += 1

    def model_policy(state):
        q_row = agent.R[state, :] + agent.gamma * agent.P[state, :, :].dot(agent.U)
        n_row = agent.Nsa[state, :]
        adjusted = np.where(n_row < agent.Ne, agent.R_plus, q_row)
        return int(np.argmax(adjusted))

    return model_policy, agent, (step_counts, returns_history)

class QLearner:
    """
    Standard Q-learning + optimistic exploration.
    """

    def __init__(self, nS, nA, gamma=0.99, alpha=0.05, R_plus=5.0, Ne=10, tie_break_random=True):
        self.nS, self.nA = nS, nA
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.R_plus = float(R_plus)
        self.Ne = int(Ne)
        self.tie_break_random = bool(tie_break_random)

        self.Q = np.zeros((nS, nA))
        self.Nsa = np.zeros((nS, nA), dtype=np.int64)

        self.s = None
        self.a = None

    def start_episode(self, s0):
        self.s = int(s0)
        self.a = self._select_action(self.s)
        return self.a

    def step(self, percept):
        """
        Standard Q-learning update + optimistic exploration action selection.
        """
        s_prime, r_prime, done = percept
        s_prime = int(s_prime)

        s = self.s
        a = self.a

        # Count visit
        self.Nsa[s, a] += 1

        # Q-learning target
        if done:
            target = r_prime
        else:
            target = r_prime + self.gamma * np.max(self.Q[s_prime, :])

        # Update Q
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

        if done:
            self.s = None
            self.a = None
            return None

        self.s = s_prime
        a_prime = self._select_action(self.s)
        self.a = a_prime
        return a_prime

    def _select_action(self, s):
        """
        f(u,n) = R_plus if n < Ne else u
        """
        s = int(s)
        q_row = self.Q[s, :]
        n_row = self.Nsa[s, :]

        adjusted = np.where(n_row < self.Ne, self.R_plus, q_row)

        if self.tie_break_random:
            max_val = np.max(adjusted)
            candidates = np.flatnonzero(np.isclose(adjusted, max_val))
            return int(np.random.choice(candidates))
        return int(np.argmax(adjusted))



def run_ql(env, steps=1000, eval_every=10, deterministic=False, success_rate=1, **agent_kwargs):
    nS, nA = env.observation_space.n, env.action_space.n
    agent = QLearner(nS=nS, nA=nA, **agent_kwargs)

    returns_history = []
    step_counts = []
    step_count = 0
    episode_count = 0

    while step_count < steps:
        obs, _ = env.reset(seed=episode_count)
        s = obs
        a = agent.start_episode(s)
        done = False
        episode_return = 0

        while not done and step_count < steps:
            obs2, r, terminated, truncated, _ = env.step(a)
            s2 = obs2
            done = terminated or truncated
            episode_return += r
            step_count += 1

            a = agent.step((s2, r, done))

            if step_count % eval_every == 0:
                avg_return = evaluate(make_env,
                                      lambda s: int(np.argmax(agent.Q[s])),
                                      episodes=50,
                                      deterministic=deterministic,
                                      success_rate=success_rate)
                returns_history.append(avg_return)
                step_counts.append(step_count)

        episode_count += 1

    def greedy(s):
        return int(np.argmax(agent.Q[s]))
    return greedy, agent, (step_counts, returns_history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ql","active_adp","plot"], required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--eval_episodes", type=int, default=50)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--success_rate", type=float, default=1)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()

    deterministic = args.deterministic

    if args.algo == "plot":
        plot_learning_curves(steps=args.steps,
                             deterministic=deterministic,
                             success_rate=args.success_rate)
    else:
        if args.algo == "active_adp":
            greedy, _, _ = run_active_adp(
                make_env(deterministic=deterministic,
                         success_rate=args.success_rate),
                steps=args.steps,
                deterministic=deterministic,
                success_rate=args.success_rate
            )
        elif args.algo == "ql":
            greedy, _, _ = run_ql(
                make_env(deterministic=deterministic,
                         success_rate=args.success_rate),
                steps=args.steps,
                deterministic=deterministic,
                success_rate=args.success_rate
            )

        avg = evaluate(make_env, greedy,
                       episodes=args.eval_episodes,
                       deterministic=deterministic,
                       success_rate=args.success_rate)

        print(f"[{args.algo.upper()}] Avg return over {args.eval_episodes} episodes: {avg:.3f}")

        if args.record:
            record_video(greedy,
                         deterministic=deterministic,
                         episodes=1,
                         algo_name=args.algo)
        if args.live:
            live_animation(greedy,
                           deterministic=deterministic,
                           steps=150)


if __name__ == "__main__":
    main()
