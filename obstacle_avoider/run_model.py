import numpy as np
import cv2 as cv
from Agent import Agent
from keras.models import load_model
from gym.envs.box2d.car_racing import CarRacing
from collections import deque


if __name__ == '__main__':
    agent = Agent()
    agent.epsilon = 0

    agent.training_model = load_model('models/Model_Eps0.99_strict_1000')

    env = CarRacing()

    while True:
        state = env.reset()
        state = cv.cvtColor(state, cv.COLOR_BGR2GRAY)
        state = state / 255.0
        total_reward = 0
        state_queue = deque([state] * 4, maxlen=4)

        done = False
        while True:
            env.render()
            old_states_array = np.array(state_queue)
            old_states = np.moveaxis(old_states_array, 0, -1)

            action = agent.get_action(old_states)
            new_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            new_state = cv.cvtColor(new_state, cv.COLOR_BGR2GRAY)
            new_state = new_state / 255.0
            state_queue.append(new_state)

            if done:
                break
