from be.kdg.rl.agent.agent import TabularAgent, Agent
from be.kdg.rl.environment.openai import FrozenLakeEnvironment
from be.kdg.rl.learning.tabular.qlearning import Qlearning, NStepQlearning, MonteCarloLearning

if __name__ == "__main__":
    # example use of the code base
    environment = FrozenLakeEnvironment()

    # # create an Agent that uses Qlearning Strategy
    agent: Agent = TabularAgent(environment, Qlearning(environment))

    # create an Agent that uses NStepQlearning Strategy
    # agent: Agent = TabularAgent(environment, NStepQlearning(environment, 5))

    # # create an Agent that uses MonteCarlolearning Strategy
    # agent: Agent = TabularAgent(environment, MonteCarloLearning(environment))

    agent.train()

