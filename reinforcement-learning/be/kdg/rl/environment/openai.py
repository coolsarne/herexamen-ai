from abc import ABC

import pygame
import gym as gym
from gym.wrappers import TimeLimit

from be.kdg.rl.environment.environment import Environment


class OpenAIGym(Environment, ABC):
    """
    Superclass for all kinds of OpenAI environments
    Wrapper for all OpenAI Environments
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        # self._env: TimeLimit = gym.make(name, render_mode="human", is_slippery=True)
        # self._env: TimeLimit = gym.make(name, is_slippery=True)
        # self._env: TimeLimit = gym.make(name, is_slippery=False)

        # self._env: TimeLimit = gym.make(name, render_mode="human")
        self._env: TimeLimit = gym.make(name, is_slippery=False)

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def render(self):
        self._env.render()

    def close(self) -> None:
        self._env.close()

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def n_actions(self):
        return self._env.action_space.n

    @property
    def state_size(self):
        if self.isdiscrete:
            return self._env.observation_space.n
        else:
            return self._env.observation_space.shape[0]

    @property
    def isdiscrete(self) -> bool:
        return hasattr(self._env.observation_space, 'n')

    @property
    def name(self) -> str:
        return self._name


class FrozenLakeEnvironment(OpenAIGym):

    def __init__(self) -> None:
        super().__init__(name="FrozenLake-v1")

    @property
    def steps_solved(self) -> int:
        return -1  # Not used for this environment


class CartPoleEnvironment(OpenAIGym):

    def __init__(self) -> None:
        super().__init__(name="CartPole-v1")

    @property
    def steps_solved(self) -> int:
        return 195
