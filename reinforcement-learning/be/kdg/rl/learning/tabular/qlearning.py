import numpy as np

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.tabular.tabular_learning import TabularLearner


class Qlearning(TabularLearner):

    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)

    def learn(self, episode: Episode):
        p = episode.percepts(1)[0]
        s = p.state
        a = p.action
        r = p.reward
        s_prime = p.next_state

        if r != 0:
            print("Gift found!")

        self.q_values[s, a] = self.q_values[s, a] + self.α * (r + self.γ * np.max(self.q_values[s_prime, :]) - self.q_values[s, a])

        super().learn(episode)


class NStepQlearning(TabularLearner):

    def __init__(self, environment: Environment, n: int, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)
        self.n = n  # maximum number of percepts before learning kicks in
        self.percepts = []  # this will buffer the percepts

    def learn(self, episode: Episode):
        # Do we have enough Percepts in the Episode E?
        if episode.size >= self.n:
            for percept in episode.percepts(self.n):
                s = percept.state
                a = percept.action
                r = percept.reward
                s_prime = percept.next_state

                if r != 0:
                    print("Gift found!")

                # q(s, a) ← q(s, a) − α · (q(s, a) −[r(s, a) + γ · maxa′(q(s′, a′))])
                q = self.q_values[s, a]
                max = np.max(self.q_values[s_prime, :])

                self.q_values[s, a] = q - self.α * (q - (r + self.γ * max))

        # call EVALUATE and IMPROVE in the superclass
        super().learn(episode)


class MonteCarloLearning(TabularLearner):
    def __init__(self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)
        self.percepts = []  # this will buffer the percepts

    def learn(self, episode: Episode):
        # Do we have enough Percepts in the Episode E?
        for percept in episode.percepts(episode.size):
            s = percept.state
            a = percept.action
            r = percept.reward
            s_prime = percept.next_state

            if r != 0:
                print("Gift found!")

            q = self.q_values[s, a]
            max = np.max(self.q_values[s_prime, :])

            self.q_values[s, a] = q - self.α * (q - (r + self.γ * max))

        # call EVALUATE and IMPROVE in the superclass
        super().learn(episode)
