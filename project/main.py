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
    print(f"Table Q invalide ({e}), création d'une nouvelle table.")
    Q = np.zeros((n_states, n_actions))


alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995

def choose_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

n_episodes = 50000

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action = choose_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
        if truncated:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 5000 == 0:
        print(f"Épisode {episode}/{n_episodes} | epsilon={epsilon:.3f}")
        np.save("q_table.npy", Q)

print("Entraînement terminé.")
np.save("q_table.npy", Q)
