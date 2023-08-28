import math

from be.kdg.rl.agent.percept import Percept
from be.kdg.rl.environment.markovdecisionprocess import MarkovDecisionProcess
from be.kdg.rl.environment.openai import FrozenLakeEnvironment, OpenAIGym
from prettytable import PrettyTable

if __name__ == '__main__':

    # setup
    environment = FrozenLakeEnvironment()
    # environment.action_space.seed(42)

    observation, info = environment.reset()

    mdp = MarkovDecisionProcess(environment)

    # run
    for i in range(100_000):
        action = environment.action_space.sample()

        new_observation, reward, terminated, truncated, info = environment.step(action)

        if terminated or truncated:
            new_observation, info = environment.reset()

        percept = Percept((observation, action, reward, new_observation, terminated))

        mdp.update(percept)
        observation = new_observation
    environment.close()

    # print mdp model
    actions = ["To State", "LEFT", "DOWN", "RIGHT", "UP"]

    for i in range(16):
        print("From State: S", (i + 1))
        firstState = mdp.P[i]
        t = PrettyTable(actions)
        for j in range(16):
            row = firstState[j]
            # if not (row[0] == 0 and row[1] == 0 and row[2] == 0 and row[3] == 0):
            t.add_row([j + 1, math.floor(row[0] * 10000) / 100, math.floor(row[1] * 10000) / 100,
                       math.floor(row[2] * 10000) / 100,
                       math.floor(row[3] * 10000) / 100])

        print(t)
