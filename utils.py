import os
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import random


# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def make_env(deterministic=False, render_mode=None, success_rate=1):
    return gym.make("FrozenLake-v1", is_slippery=not deterministic, success_rate=success_rate, render_mode=render_mode)

def record_video(act_fn, deterministic=False, out_dir="./videos", episodes=1, algo_name="unknown"):
    import datetime
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Create timestamp and filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    env_type = "det" if deterministic else "stoch"
    name_prefix = f"frozenlake_{algo_name}_{env_type}_{timestamp}"
    
    venv = gym.wrappers.RecordVideo(
        make_env(deterministic=deterministic, render_mode="rgb_array"),
        video_folder=out_dir, name_prefix=name_prefix)
    for ep in range(episodes):
        obs,_=venv.reset(seed=456+ep); s=obs; done=False
        while not done:
            a=act_fn(s)
            obs,r,done,trunc,_=venv.step(a)
            s=obs; done = done or trunc
    venv.close()
    print(f"Video saved to: {out_dir}/{name_prefix}-episode-0.mp4")

def live_animation(act_fn, deterministic=False, steps=150):
    env=make_env(deterministic=deterministic, render_mode="human")
    obs,_=env.reset(seed=999); s=obs
    for t in range(steps):
        a=act_fn(s)
        obs,r,done,trunc,_=env.step(a)
        s=obs
        if done or trunc: break
    env.close()

def evaluate(make_env_fn, act_fn, episodes=50, deterministic=False, success_rate=1):
    total = 0.0
    eval_env = make_env_fn(deterministic=deterministic, render_mode=None, success_rate=success_rate)
    try:
        for ep in range(episodes):
            obs, _ = eval_env.reset(seed=123 + ep)
            s = obs
            done = False
            ret = 0.0
            while not done:
                a = act_fn(s)
                obs, r, terminated, truncated, _ = eval_env.step(a)
                s = obs
                done = terminated or truncated
                ret += r
            total += ret
    finally:
        eval_env.close()
    return total / episodes

def plot_learning_curves(steps=50000, deterministic=False, success_rate=1):
    """Run both algorithms and plot their learning curves"""
    # Import the training functions from run.py
    from run import run_ql, run_active_adp
    
    env_ql  = make_env(deterministic, success_rate=success_rate)
    ql_greedy, ql_agent, ql_data = run_ql(env_ql, steps=steps, eval_every=10, deterministic=deterministic, success_rate=success_rate)
    env_ql.close()

    env_adp = make_env(deterministic, success_rate=success_rate)
    adp_greedy, adp_agent, adp_data = run_active_adp(env_adp, steps=steps, eval_every=10, deterministic=deterministic, success_rate=success_rate)
    env_adp.close()
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    ql_steps, ql_returns = ql_data
    adp_steps, adp_returns = adp_data
    
    plt.plot(ql_steps, ql_returns, 'b-', linewidth=2, label='Q-Learning', marker='o', markersize=4)
    plt.plot(adp_steps, adp_returns, 'r-', linewidth=2, label='Active ADP', marker='s', markersize=4)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Average Return (50 episodes)', fontsize=12)
    plt.title(f'Learning Curves Comparison - Frozen Lake ({"Deterministic" if deterministic else "Stochastic"})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, steps)
    plt.ylim(0, 1.1)
    
    # Add final performance annotations
    final_ql = ql_returns[-1] if ql_returns else 0
    final_adp = adp_returns[-1] if adp_returns else 0
    
    plt.annotate(f'QL Final: {final_ql:.3f}', 
                xy=(steps, final_ql), xytext=(steps*0.7, final_ql + 0.1),
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                fontsize=10, color='blue')
    
    plt.annotate(f'Active ADP Final: {final_adp:.3f}', 
                xy=(steps, final_adp), xytext=(steps*0.7, final_adp - 0.1),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, color='red')
    
    # Create filename with timestamp and success_rate for stochastic
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if deterministic:
        filename = f'frozen_lake_learning_curves_det_{timestamp}.png'
    else:
        # For stochastic, include success_rate if provided
        if success_rate is not None:
            filename = f'frozen_lake_learning_curves_stoch_sr{success_rate}_{timestamp}.png'
        else:
            filename = f'frozen_lake_learning_curves_stoch_{timestamp}.png'
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved as: {filename}")
    
    print(f"\nFinal Performance:")
    print(f"Q-Learning: {final_ql:.3f}")
    print(f"Active ADP: {final_adp:.3f}")
    
    return ql_greedy, adp_greedy