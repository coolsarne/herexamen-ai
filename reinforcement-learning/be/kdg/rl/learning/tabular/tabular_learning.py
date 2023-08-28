from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.learningstrategy import LearningStrategy


class TabularLearner(LearningStrategy):
    """
    A tabular learner implements a tabular method such as Q-Learning, N-step Q-Learning, ...
    """
    π: ndarray
    v_values: ndarray
    q_values: ndarray

    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        super().__init__(environment, λ, γ, t_max)
        # learning rate
        self.α = α

        # policy table
        self.π = np.full((self.env.state_size, self.env.n_actions), fill_value=1 / self.env.n_actions)

        # state value table
        self.v_values = np.full((self.env.state_size,), 0.0)

        # state-action table
        self.q_values = np.full((self.env.state_size, self.env.n_actions), 0.0)

    @abstractmethod
    def learn(self, episode: Episode):
        # subclasses insert their implementation at this point
        # see for example be\kdg\rl\learning\tabular\qlearning.py
        self.evaluate()
        self.improve()
        super().learn(episode)

    def on_learning_start(self):
        self.t = 0

    def next_action(self, s: int):
        a = np.arange(0, self.env.n_actions)
        p = self.π[s]
        p /= p.sum()  # normalize
        return np.random.choice(a, p=p)

    def evaluate(self):
        for s in range(self.env.state_size):
            self.v_values[s] = np.max(self.q_values[s])

    def improve(self):
        for s in range(self.env.state_size):
            for a in range(self.env.n_actions):
                q = self.q_values[s]
                ax = np.random.choice(np.flatnonzero(q == q.max()))
                if ax == a:
                    self.π[s, a] = 1 - self.ε + self.ε / self.env.n_actions
                else:
                    self.π[s, a] = self.ε / self.env.n_actions
        self.decay()

    def quiver_plot(self):
        # https: // www.geeksforgeeks.org / quiver - plot - in -matplotlib /
        # Creating arrow
        x_pos = 0
        y_pos = 3

        fig, axis = plt.subplots(figsize=(6, 3))
        # Creating plot
        for s in range(self.env.state_size):
            ax = np.argmax(self.π[s], axis=0)
            if x_pos < 4:
                if ax == 0:
                    axis.quiver(x_pos, y_pos, -1, 0)
                if ax == 1:
                    axis.quiver(x_pos, y_pos, 0, -1)
                if ax == 2:
                    axis.quiver(x_pos, y_pos, 1, 0)
                if ax == 3:
                    axis.quiver(x_pos, y_pos, 0, 1)
                x_pos += 1
            if x_pos == 4:
                x_pos = 0
                y_pos -= 1

        # Show plot
        plt.show()

    def save_network(self):
        pass  # Ignore
