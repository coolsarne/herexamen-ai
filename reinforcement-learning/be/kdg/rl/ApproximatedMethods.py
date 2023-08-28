import os

from be.kdg.rl.agent.agent import Agent, ApproximateAgent, PlayerAgent
from be.kdg.rl.environment.openai import CartPoleEnvironment
from be.kdg.rl.learning.approximate.deep_qlearning import DeepQLearning

from tensorflow import keras

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    environment = CartPoleEnvironment()
    training: bool = True
    model_path = os.path.join(local_dir, 'models')

    # Learning Parameters (Using defaults for the rest)
    ddqn: bool = False  # Use Double Deep Q-Learning.
    n_episodes = 150  # Number of episodes.
    batch_size = 4  # Size of batch taken from replay buffer to train network with.
    c = 10  # Copy q1 model weights every C steps to q2 model.

    # Train the agent.
    if training:
        agent: Agent = ApproximateAgent(environment, DeepQLearning(environment, batch_size=batch_size, C=c, ddqn=ddqn,
                                                                   model_path=model_path),
                                        n_episodes=n_episodes)
        agent.train()

    # Use a saved model to let the agent play.
    else:
        model = keras.models.load_model(os.path.join(local_dir, 'models/q1'))
        player_agent = PlayerAgent(environment, model)
        player_agent.run()
