import gymnasium as gym
import numpy as np
import time

Q = np.load("q_table.npy")
env = gym.make("Taxi-v3", render_mode="human")

n_success = 0
n_episodes = 10

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(Q[state])
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        time.sleep(0.1)
        if truncated:
            break

    if total_reward > 0:
        n_success += 1
    print(f"Épisode {episode + 1} | Récompense totale : {total_reward}")

print(f"\nTaux de succès : {n_success}/{n_episodes}")
env.close()
