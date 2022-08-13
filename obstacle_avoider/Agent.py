import random
from collections import deque

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


class Agent:
    def __init__(self):
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
        self.DECAY_RATE = 0.95
        self.MIN_EPSILON = 0.1

    def define_model(self):
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
        rand = random.random()
        if rand < self.epsilon:
            return random.choice(self.actions)
        best_action_index = np.argmax(self.training_model.predict(state.reshape(-1, *state.shape))[0])
        return self.actions[best_action_index]

    def train(self, sample):
        states = []
        new_states = []
        for batch_item in sample:
            states.append(batch_item[0])
            new_states.append(batch_item[1])

        training = self.training_model.predict(np.array(states), self.BATCH_SIZE)
        target = self.target_model.predict(np.array(new_states), self.BATCH_SIZE)

        # calculates max q value for each batch entry returned by prediction in target model
        max_q_value_per_item = np.amax(target, axis=1)

        all_q_values = [None] * len(sample)

        for i, batch_item in enumerate(sample):
            action_index = batch_item[2]
            done = batch_item[3]
            reward = batch_item[4]
            truncated = batch_item[5]
            if done or truncated:
                q_value = reward
            else:
                q_value = reward + 0.95 * max_q_value_per_item[i]

            all_q_values[i] = training[i]
            all_q_values[i][action_index] = q_value

        return self.training_model.fit(np.array(states), np.array(all_q_values), self.BATCH_SIZE, shuffle=False)
