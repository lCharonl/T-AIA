import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v3")

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


alpha = 0.1 # Taux d'apprentissage
gamma = 0.99 # Facteur récompense
epsilon = 1.0 # Taux d'exploration


epsilon_min = 0.01
epsilon_decay = 0.9995

def choose_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

n_episodes = 50000

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
