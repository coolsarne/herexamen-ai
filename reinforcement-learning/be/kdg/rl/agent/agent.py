import time
from abc import abstractmethod

import numpy as np

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.agent.percept import Percept
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.environment.openai import CartPoleEnvironment
from be.kdg.rl.learning.approximate.deep_qlearning import DeepQLearning
from be.kdg.rl.learning.learningstrategy import LearningStrategy
from be.kdg.rl.learning.tabular.tabular_learning import TabularLearner
import matplotlib.pyplot as plt

from tensorflow import keras


class Agent:

    def __init__(self, environment: Environment, learning_strategy: LearningStrategy, n_episodes=10000):
        super().__init__()
        self.env = environment
        self.learning_strategy = learning_strategy
        self.episodes: [Episode] = []
        self.n_episodes = n_episodes  # total episodes
        self.episode_count = 0

    @abstractmethod
    def train(self) -> None:
        pass

    @property
    def done(self):
        return self.episode_count > self.n_episodes


class TabularAgent(Agent):

    def __init__(self, environment: Environment, learning_strategy: TabularLearner, n_episodes=1_000) -> None:
        super().__init__(environment, learning_strategy, n_episodes)

    def train(self) -> None:
        super(TabularAgent, self).train()

        # as longs as the agents hasn't reached the maximum number of episodes
        while not self.done:

            # start a new episode
            episode = Episode(self.env)
            self.episodes.append(episode)
            # initialize the start state
            state, _ = self.env.reset()
            # reset the learning strategy
            self.learning_strategy.on_learning_start()

            # while the episode isn't finished by length
            while not self.learning_strategy.done():

                # learning strategy (policy) determines next action to take
                action = self.learning_strategy.next_action(state)
                # agent observes the results of his action
                # step method returns a tuple with values (s', r, terminated, truncated, info)
                t, r, terminated, truncated, info = self.env.step(action)

                # render environment (don't render every step, only every X-th, or at the end of the learning process)
                # self.env.render()

                # create Percept object from observed values state,action,r,s' (SARS') and terminate flag, but
                # ignore values truncated and info
                percept = Percept((state, action, r, t, terminated))

                # add the newly created Percept to the Episode
                episode.add(percept)

                # update Agent's state
                state = percept.next_state

                # learn from Percepts in Episode
                self.learning_strategy.learn(episode)

                # learn from one or more Percepts in the Episode
                # self.learning_strategy.learn(episode)

                # update state
                state = percept.next_state

                # break if episode is over
                if percept.done:
                    break

            # end episode
            self.episode_count += 1
            if self.episode_count % 100 == 0:
                print("Done with episode ", self.episode_count)

        self.learning_strategy.quiver_plot()
        self.env.close()


class ApproximateAgent(Agent):

    def __init__(self, environment: Environment, learning_strategy: DeepQLearning, n_episodes=1_000) -> None:
        super().__init__(environment, learning_strategy, n_episodes)

    def train(self) -> None:
        super(ApproximateAgent, self).train()
        history_rewards = []
        history_episode_epsilon = []
        history_episode_time = []
        while not self.done:

            # Start a new episode. (A "session" where we try to balance the pole for max_steps (195) steps)
            episode = Episode(self.env)
            self.episodes.append(episode)

            # Prepare environment.
            state, _ = self.env.reset()
            self.learning_strategy.on_learning_start()
            start_time = time.time()

            # Don't waste resources, when max steps is reached consider challenge solved.
            while episode.compute_rewards() <= self.env.steps_solved:

                # Get next action from learning strategy. This can be random or taken from the neural network based on
                # the value of ϵ that will decay over the learning time.
                action = self.learning_strategy.next_action(state)

                # Perform a step on the environment.
                t, r, terminated, truncated, info = self.env.step(action)

                # Build percept from step returns.
                percept = Percept((state, action, r, t, terminated))

                # Add percept to episode percepts.
                episode.add(percept)

                # Call learn method from learning strategy.
                self.learning_strategy.learn(episode)

                # Update state.
                state = percept.next_state

                # Break out of this loop if percept indicates episode should be over.
                if percept.done:
                    break

            # End of episode
            self.episode_count += 1
            print("Done with episode ", self.episode_count, " with reward ", episode.compute_rewards())

            # Add info to history arrays.
            history_rewards.append(episode.compute_rewards())
            history_episode_epsilon.append(self.learning_strategy.ε)
            elapsed_time = time.time() - start_time
            history_episode_time.append(int(elapsed_time))

            # Notify console that the challenge was solved in this episode.
            if episode.compute_rewards() >= self.env.steps_solved:
                print("Solved challenge in episode ", self.episode_count)
                # Save the model of this episode.
                self.learning_strategy.save_network()

        # Plot rewards over episodes.
        self.plot_results(history_rewards, history_episode_epsilon, history_episode_time)

        # Close environment.
        self.env.close()

    @staticmethod
    def plot_results(history_rewards, history_episode_epsilon, history_episode_time):
        fig, axs = plt.subplots(3, sharex=True)
        plt.xlabel('Episode')
        axs[0].set_title('Rewards (Steps)')
        axs[0].plot(history_rewards)
        axs[1].set_title('Epsilon (Exploration decay)')
        axs[1].plot(history_episode_epsilon)
        axs[2].set_title('Episode time in seconds.')
        axs[2].plot(history_episode_time)
        fig.tight_layout()
        plt.show()


class PlayerAgent:

    def __init__(self, env: CartPoleEnvironment, model: keras.Model):
        self.env = env
        self.model = model

    def run(self):
        while True:

            # Prepare environment.
            state, _ = self.env.reset()

            while True:

                # Get next action
                s = np.reshape(state, [1, self.env.state_size])
                action = np.argmax(self.model.predict(s, verbose=0))

                # Step
                next_state, _, terminated, _, _ = self.env.step(action)
                state = next_state

                # Break out of this loop if percept indicates episode should be over.
                if terminated:
                    break
