import random
from collections import deque

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


class Agent:
    """DQN Agent that is trained to stay on the road and avoid obstacles
    """
    def __init__(self):
        """Initialize Agent

        Attributes:
            self.actions: The action space in the form (steering, gas, brake), where steering [-1, 1], gas [0, 1] and
                brake [0, 1]
            self.training_model: The model that is trained and used to choose actions.
            self.target_model: The model that is used to evaluate chosen actions.
            self.BUFFER_SIZE: The size of the replay memory buffer.
            self.buffer: The replay memory buffer.
            self.BATCH_SIZE: The size of the batches that get fed to the model.
            self.epsilon: The starting value of epsilon.
            self.DECAY_RATE: The rate at which epsilon decays.
            self.MIN_EPSILON: The minimum value epsilon can have.
        """
        self.actions = [
            (-1, 1, 0), (0, 1, 0), (1, 1, 0),
            (-1, 0, 0.3), (0, 0, 0.3), (1, 0, 0.3),
            (-1, 0, 0), (0, 0, 0), (1, 0, 0)
        ]
        self.training_model = self.define_model()
        self.target_model = self.define_model()
        self.BUFFER_SIZE = 2000
        self.buffer = deque(maxlen=self.BUFFER_SIZE)
        self.BATCH_SIZE = 64
        self.epsilon = 1
        self.DECAY_RATE = 0.999
        self.MIN_EPSILON = 0.1

    def define_model(self):
        """Defines and creates a model

        Returns:
            The created model
        """
        model = keras.Sequential(
            [
                layers.Conv2D(4, kernel_size=(7, 7), strides=(3, 3), input_shape=(96, 96, 4), activation='relu'),
                layers.MaxPooling2D(pool_size=(3, 3)),
                layers.Conv2D(8, kernel_size=(5, 5), activation='relu'),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(256),
                layers.Dense(len(self.actions)),
            ]
        )
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def get_action(self, state):
        """Chooses a random action with probability self.epsilon or the currently best action with probability 1-self.epsilon

        Args:
            state: The current state

        Returns:
            The index of the chosen action
        """
        rand = random.random()
        if rand < self.epsilon:
            return random.choice(self.actions)
        best_action_index = np.argmax(self.training_model.predict(state.reshape(-1, *state.shape))[0])
        return self.actions[best_action_index]

    def train(self, sample):
        """Trains the model and evaluates how good the chosen action is

        Args:
            sample: A batch of size self.BATCH_SIZE of experiences

        Returns:
            Training metrics (accuracy and loss)
        """
        states = []
        new_states = []
        for batch_item in sample:
            states.append(batch_item[0])
            new_states.append(batch_item[1])

        # predict q values for every action for every state in the batch
        training = self.training_model.predict(np.array(states), self.BATCH_SIZE, verbose=0)

        # predict q values for every action for every new_state in the batch
        target = self.target_model.predict(np.array(new_states), self.BATCH_SIZE, verbose=0)

        # calculates max q value for each batch entry returned by prediction in target model
        max_q_value_per_item = np.amax(target, axis=1)

        for i, batch_item in enumerate(sample):
            action_index = batch_item[2]
            done = batch_item[3]
            reward = batch_item[4]
            truncated = batch_item[5]
            if done or truncated:
                q_value = reward
            else:
                q_value = reward + 0.95 * max_q_value_per_item[i]
            training[i][action_index] = q_value

        return self.training_model.fit(np.array(states), np.array(training), self.BATCH_SIZE, verbose=0)
