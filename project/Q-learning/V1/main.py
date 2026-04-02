import json
import gymnasium as gym
import numpy as np

with open("config.json") as f:
    cfg_file = json.load(f)

cfg = cfg_file["configs"][cfg_file["active"]]
print(f"Config chargée : {cfg_file['active']}")

alpha         = cfg["alpha"]
gamma         = cfg["gamma"]
epsilon       = cfg["epsilon"]
epsilon_min   = cfg["epsilon_min"]
epsilon_decay = cfg["epsilon_decay"]
n_episodes    = cfg["n_episodes"]

env = gym.make("Taxi-v3")
#env = gym.make("Taxi-v3", render_mode="human")

n_actions = env.action_space.n
n_states = env.observation_space.n

try:
    Q = np.load("q_table.npy")
    if Q.shape != (n_states, n_actions):
        raise ValueError(f"Mauvaise forme : {Q.shape}, attendu ({n_states}, {n_actions})")
    print("Table Q chargée.")
except Exception as e:
    print(f"Table Q invalide ({e}), création nouvelle table.")
    Q = np.zeros((n_states, n_actions))



def choose_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

rewards_per_episode = []
steps_per_episode = []
epsilons = []

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = choose_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
        total_reward += reward
        steps += 1
        if truncated:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)
    steps_per_episode.append(steps)
    epsilons.append(epsilon)

    if episode % 5000 == 0:
        print(f"Épisode {episode}/{n_episodes} | epsilon={epsilon:.3f}")
        np.save("q_table.npy", Q)

print("Entraînement terminé.")
np.save("q_table.npy", Q)
np.save("rewards.npy", np.array(rewards_per_episode))
np.save("steps.npy", np.array(steps_per_episode))
np.save("epsilons.npy", np.array(epsilons))
