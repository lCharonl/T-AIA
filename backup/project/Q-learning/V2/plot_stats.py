import numpy as np
import matplotlib.pyplot as plt

results = np.load("results.npy", allow_pickle=True).item()

window = 500

def moving_avg(data, w):
    return np.convolve(data, np.ones(w) / w, mode="valid")

colors = plt.cm.tab10.colors
cfg_names = list(results.keys())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Q-Learning — Taxi-v3 | Comparaison des configs", fontsize=14)

# 1. Reward par épisode
ax = axes[0, 0]
for i, name in enumerate(cfg_names):
    avg = moving_avg(results[name]["rewards"], window)
    ax.plot(avg, color=colors[i], label=name)
ax.set_title("Total Reward per Episode")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.legend()

# 2. Steps par épisode
ax = axes[0, 1]
for i, name in enumerate(cfg_names):
    avg = moving_avg(results[name]["steps"], window)
    ax.plot(avg, color=colors[i], label=name)
ax.set_title("Steps per Episode")
ax.set_xlabel("Episode")
ax.set_ylabel("Steps")
ax.legend()

# 3. Epsilon decay
ax = axes[1, 0]
for i, name in enumerate(cfg_names):
    ax.plot(results[name]["epsilons"], color=colors[i], label=name)
ax.set_title("Epsilon Decay (Exploration Rate)")
ax.set_xlabel("Episode")
ax.set_ylabel("Epsilon")
ax.legend()

# 4. Taux de succès glissant
ax = axes[1, 1]
for i, name in enumerate(cfg_names):
    successes = (results[name]["rewards"] > 0).astype(float)
    rate = moving_avg(successes, window) * 100
    ax.plot(rate, color=colors[i], label=name)
ax.set_title(f"Success Rate (%) — Moving avg ({window})")
ax.set_xlabel("Episode")
ax.set_ylabel("Success (%)")
ax.set_ylim(0, 100)
ax.legend()

plt.tight_layout()
plt.savefig("training_metrics.png", dpi=150)
plt.show()
print("Graphiques sauvegardés dans training_metrics.png")

# --- Tableau récapitulatif ---
last_n = 2000
print(f"\n=== Performance Summary (last {last_n} episodes) ===")
print(f"{'Config':<20} {'Reward':>10} {'Steps':>10} {'Success%':>10} {'Epsilon':>10}")
print("-" * 62)
for name in cfg_names:
    r = results[name]["rewards"][-last_n:]
    s = results[name]["steps"][-last_n:]
    e = results[name]["epsilons"][-1]
    print(f"{name:<20} {np.mean(r):>10.2f} {np.mean(s):>10.1f} {np.mean(r > 0)*100:>9.1f}% {e:>10.4f}")
