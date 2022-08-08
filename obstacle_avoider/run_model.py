import numpy as np
from Agent import Agent
from keras.models import load_model
from gym.envs.box2d.car_racing import CarRacing


if __name__ == '__main__':
    agent = Agent()
    model = load_model('model/model_full')
    env = CarRacing()

    while True:
        state = env.reset()

        done = False
        while True:

            env.render()
            q_values = model.predict(state.reshape(-1, *state.shape))[0]
            action = np.argmax(q_values)
            state, reward, done, truncated, _ = env.step(action)
            if done or truncated:
                break

            print(f' Q Values: [{"{:.2f}".format(q_values[0])}, {"{:.2f}".format(q_values[1])}, '
                  f'{"{:.2f}".format(q_values[2])}, {"{:.2f}".format(q_values[3])}, {"{:.2f}".format(q_values[4])}]'
                  f' \n Action taken: {action}')
