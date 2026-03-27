import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("rewards.npy")
steps = np.load("steps.npy")
epsilons = np.load("epsilons.npy")
Q = np.load("q_table.npy")

n_episodes = len(rewards)
window = 500

def moving_avg(data, w):
    return np.convolve(data, np.ones(w) / w, mode="valid")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Q-Learning — Taxi-v3 Training Metrics", fontsize=14)

# 1. Reward par épisode + moving average
ax = axes[0, 0]
ax.plot(rewards, alpha=0.3, color="steelblue", label="Reward")
ax.plot(moving_avg(rewards, window), color="steelblue", label=f"Moving avg ({window})")
ax.set_title("Total Reward per Episode")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.legend()

# 2. Steps par épisode + moving average
ax = axes[0, 1]
ax.plot(steps, alpha=0.3, color="darkorange", label="Steps")
ax.plot(moving_avg(steps, window), color="darkorange", label=f"Moving avg ({window})")
ax.set_title("Steps per Episode")
ax.set_xlabel("Episode")
ax.set_ylabel("Steps")
ax.legend()

# 3. Epsilon decay
ax = axes[1, 0]
ax.plot(epsilons, color="green")
ax.set_title("Epsilon Decay (Exploration Rate)")
ax.set_xlabel("Episode")
ax.set_ylabel("Epsilon")

# 4. Taux de succès glissant (reward > 0 = succès)
ax = axes[1, 1]
successes = (np.array(rewards) > 0).astype(float)
success_rate = moving_avg(successes, window) * 100
ax.plot(success_rate, color="crimson")
ax.set_title(f"Success Rate (%) — Moving avg ({window})")
ax.set_xlabel("Episode")
ax.set_ylabel("Success (%)")
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("training_metrics.png", dpi=150)
plt.show()
print("Graphiques sauvegardés dans training_metrics.png")

# --- Tableau récapitulatif ---
last_n = 5000
print("\n=== Performance Summary (last 5000 episodes) ===")
print(f"Mean reward     : {np.mean(rewards[-last_n:]):.2f}")
print(f"Std reward      : {np.std(rewards[-last_n:]):.2f}")
print(f"Mean steps      : {np.mean(steps[-last_n:]):.1f}")
print(f"Success rate    : {np.mean(rewards[-last_n:] > 0) * 100:.1f}%")
print(f"Final epsilon   : {epsilons[-1]:.4f}")
print(f"Q-table max val : {Q.max():.3f}")
print(f"Q-table min val : {Q.min():.3f}")
print(f"Q-table zeros   : {(Q == 0).sum()} / {Q.size}")
