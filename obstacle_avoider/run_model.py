import numpy as np
import cv2 as cv
from Agent import Agent
from keras.models import load_model
from gym.envs.box2d.car_racing import CarRacing
from collections import deque


def main():
    """Runs the given model
    """
    agent = Agent()

    # set epsilon to 0 to enable full exploitation
    agent.epsilon = 0

    agent.training_model = load_model('models/Model_Eps0.995_875.h5')

    env = CarRacing()

    while True:
        # getting the initial state, processing it and putting it four times into a deque (state_queue)
        # state_queue always keeps the last four states so that they can be fed to the agent together
        state = env.reset()
        state = cv.cvtColor(state, cv.COLOR_BGR2GRAY)
        state = state / 255.0
        state_queue = deque([state] * 4, maxlen=4)

        done = False
        while True:
            env.render()

            # preprocessing the already existing states (moving axis from (4, x, y) to (x, y, 4))
            old_states_array = np.array(state_queue)
            old_states = np.moveaxis(old_states_array, 0, -1)

            # getting the action that the agent shall perform based on the already existing states
            action = agent.get_action(old_states)

            # executing action in current state to get new state
            new_state, reward, done, truncated, _ = env.step(action)

            # processing the new state and putting the new state into the state_queue
            new_state = cv.cvtColor(new_state, cv.COLOR_BGR2GRAY)
            new_state = new_state / 255.0
            state_queue.append(new_state)

            if done:
                break


if __name__ == '__main__':
    main()
