import json
import gymnasium as gym
import numpy as np

with open("config.json") as f:
    cfg_file = json.load(f)

env = gym.make("Taxi-v3")
n_actions = env.action_space.n
n_states = env.observation_space.n

results = {}

for cfg_name, cfg in cfg_file["configs"].items():
    print(f"\n=== Entraînement : {cfg_name} ===")

    alpha         = cfg["alpha"]
    gamma         = cfg["gamma"]
    epsilon       = cfg["epsilon"]
    epsilon_min   = cfg["epsilon_min"]
    epsilon_decay = cfg["epsilon_decay"]
    n_episodes    = cfg["n_episodes"]

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
            print(f"  Épisode {episode}/{n_episodes} | epsilon={epsilon:.3f}")

    print(f"  Entraînement terminé.")
    results[cfg_name] = {
        "rewards": np.array(rewards_per_episode),
        "steps": np.array(steps_per_episode),
        "epsilons": np.array(epsilons),
    }

env.close()
np.save("results.npy", results)
print("\nRésultats sauvegardés dans results.npy")
