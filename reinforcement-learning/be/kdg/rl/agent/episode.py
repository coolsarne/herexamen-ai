from collections import deque

import numpy as np

from be.kdg.rl.agent.percept import Percept
from be.kdg.rl.environment.environment import Environment


class Episode:
    """
    A collection of Percepts forms an Episode. A Percept is added per step/time t.
    The Percept contains the state, action, reward and next_state.
    This class is INCOMPLETE
    """

    def __init__(self, env: Environment) -> None:
        self._env = env
        self._percepts: [Percept] = deque()

    def add(self, percept: Percept):
        self._percepts.append(percept)

    def percepts(self, n: int):
        """ Return n final percepts from this Episode """
        percepts = list()
        for i in range(1, n + 1):
            percepts.append(self._percepts[-i])
        return percepts

    def compute_rewards(self):
        """ Return sum of rewards for this Episode """
        rewards = []
        for percept in self._percepts:
            rewards.append(percept.reward)
        return np.sum(rewards)

    def sample(self, batch_size: int):
        """ Sample and return a random batch of Percepts from this Episode """
        if batch_size > self.size:
            pass

        np.shuffle(self._percepts)
        return self._percepts[:batch_size]

    @property
    def size(self):
        return len(self._percepts)
