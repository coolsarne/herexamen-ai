import os
import random

import tensorflow as tf
import numpy as np

from keras import Model, Input
from keras.layers import Dense
from keras.optimizers import Adam

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.learningstrategy import LearningStrategy


class DeepQLearning(LearningStrategy):
    """
    Two neural nets q1 en q2 are trained together and used to predict the best action.
    These nets are denoted as Q1 and Q2 in the pseudocode.
    This class is INCOMPLETE.
    """
    q1: Model  # keras NN
    q2: Model  # keras NN
    replay_memory: []
    model_path: str

    def __init__(self, environment: Environment, batch_size: int, C: int, ddqn=False, λ=0.001, γ=0.99, t_max=195,
                 replay_memory_size: int = 1000, model_path='/models') -> None:
        super().__init__(environment, λ, γ, t_max)
        self.batch_size = batch_size
        self.ddqn = ddqn  # are we using double deep q learning network?
        # neural networks with weights
        self.q1 = self.build_nn()
        self.q2 = self.build_nn()
        # counter is used for updating weights of q2 in learn from batch method
        self.counter = 0
        # Copy weight every C steps
        self.C = C
        # size of replay memory list
        self.replay_memory_size = replay_memory_size
        # Replay memory, stores percepts used for training.
        self.replay_memory = []
        self.model_path = model_path

    # build neural network to use instead of policy table
    def build_nn(self) -> Model:
        inputs = Input(shape=(self.env.state_size,))
        x = Dense(96, activation='relu')(inputs)
        x = Dense(96, activation='relu')(x)
        outputs = Dense(self.env.action_space.n, activation="linear")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

        return model

    def on_learning_start(self):
        self.t = 0
        self.counter = 0

    def next_action(self, state):
        """ Neural net decides on the next action to take """
        # exploration - exploitation mechanism
        # if probability of exploring is bigger than random value -> max prediction of q1
        if random.random() < self.ε:
            return self.env.action_space.sample()
        else:
            s = np.reshape(state, [1, self.env.state_size])
            return np.argmax(self.q1.predict(s, verbose=0))

    def add_percept_to_replay_memory(self, percept):
        # append replay memory if size is smaller than preferred replay memory size
        if len(self.replay_memory) < self.replay_memory_size:
            self.replay_memory.append(percept)
        # else set the percept with a random index of the replay memory
        else:
            self.replay_memory[random.randint(0, self.replay_memory_size - 1)] = percept

    def learn(self, episode: Episode):
        """ Sample batch from Episode and train NN on sample"""
        # store last percept from episode in replay memory R
        self.add_percept_to_replay_memory(episode.percepts(1)[0])

        # If there are enough percepts in replay memory
        if len(self.replay_memory) >= self.batch_size:
            # sample random percepts from replay memory
            sample_batch = random.sample(self.replay_memory, self.batch_size)
            random.shuffle(sample_batch)
            # learn from sample P
            self.learn_from_batch(sample_batch)
            super().learn(episode)
            # Reduce epsilon ε, because we need less and less exploration as time progresses
            self.decay()

    def learn_from_batch(self, p: list):
        training_set = self.build_training_set(p)
        self.train_network(training_set)
        self.counter += 1
        # copy weights from q1 tinto weights of q2 every C times
        if self.counter % self.C == 0:
            self.q2.set_weights(self.q1.get_weights())

    def build_training_set(self, percepts: list):
        """ Build training set from episode """
        # empty trainingsset
        training_set = []
        # for every percept in sample of percepts from replay memory
        for p in percepts:

            s = np.reshape(p.state, [1, self.env.state_size])
            a = p.action
            r = p.reward
            s_next = np.reshape(p.next_state, [1, self.env.state_size])
            # predict q-values for state s (both numpy arrays)
            qs = self.q1.predict(s, verbose=0)
            if self.ddqn:
                # DDQL: training vectors created by both q1 and q2
                # best action as per q1
                a_star = np.argmax(self.q1.predict(s_next, verbose=0))
                q_star = self.q2.predict(s_next, verbose=0)[0][a_star]
            else:
                # DQL: training vectors completely determined by q2
                q_star_predictions = self.q2.predict(s_next, verbose=0)
                # max q-value according to q2
                q_star = np.max(q_star_predictions)

            if p.done:
                qs[0][a] = r
            else:
                # is percept is not done, take discount rate into account to calculate reward
                qs[0][a] = r + self.γ * q_star

            training_set.append((s, qs))

        return training_set

    def train_network(self, training_set):
        """ Train neural net on training set """
        # train on each state in training set
        for s, qs in training_set:
            # train neural net on training set
            self.q1.fit(s, qs, epochs=1, verbose=0)

    def save_network(self):
        self.q1.save(os.path.join(self.model_path, "q1"))
        self.q2.save(os.path.join(self.model_path, "q2"))

    def quiver_plot(self):
        pass  # Ignore
