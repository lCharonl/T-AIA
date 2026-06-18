from .base_agent import BaseAgent, StepEvent
from .brute_force import BruteForceAgent
from .q_learning import QLearningAgent
from .sarsa import SarsaAgent
from .dqn import DQNAgent

ALGO_REGISTRY = {
    "Brute Force": BruteForceAgent,
    "Q-Learning": QLearningAgent,
    "SARSA": SarsaAgent,
    "Deep Q-Learning": DQNAgent,
}

__all__ = [
    "BaseAgent",
    "StepEvent",
    "BruteForceAgent",
    "QLearningAgent",
    "SarsaAgent",
    "DQNAgent",
    "ALGO_REGISTRY",
]
