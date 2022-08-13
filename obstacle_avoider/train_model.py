import random
import cv2 as cv
import numpy as np
from collections import deque
from Agent import Agent
from gym.envs.box2d.car_racing import CarRacing


def main():
    episodes = 20000
    env = CarRacing()
    agent = Agent()

    for episode in range(1, episodes + 1):
        print(f'Episode {episode}')
        print('###########################################################')
        state = env.reset()
        state = cv.cvtColor(state, cv.COLOR_BGR2GRAY)
        state = state / 255.0
        state_list = [state]
        state_queue = deque(state_list * 4, maxlen=4)

        done = False
        update_steps = 0
        episode_reward = 0
        negative_reward_streak = 0

        while not done:
            env.render()
            old_states_array = np.array(state_queue)
            old_states = np.moveaxis(old_states_array, 0, -1)
            action = agent.get_action(old_states)

            print(episode_reward)
            new_state, reward, done, truncated, _ = env.step(action)

            new_state = cv.cvtColor(new_state, cv.COLOR_BGR2GRAY)
            new_state = new_state / 255.0
            state_queue.append(new_state)
            new_states_array = np.array(state_queue)
            next_states = np.moveaxis(new_states_array, 0, -1)

            if reward < 0:
                negative_reward_streak += 1
            else:
                negative_reward_streak = 0
            episode_reward += reward
            agent.buffer.append((old_states, next_states, agent.actions.index(action), done, reward, truncated))

            if len(agent.buffer) >= agent.BATCH_SIZE:
                sample = random.sample(agent.buffer, agent.BATCH_SIZE)
                agent.train(sample)
                update_steps += 1
                if update_steps == 15:
                    agent.target_model.set_weights(agent.training_model.get_weights())
                    update_steps = 0
            if truncated or episode_reward < -50 or negative_reward_streak >= 25:
                break

        agent.epsilon = agent.epsilon * agent.DECAY_RATE
        if agent.epsilon < agent.MIN_EPSILON:
            agent.epsilon = agent.MIN_EPSILON
        if episode % 100 == 0:
            agent.training_model.save(f'model/model_episodes{i}')

    agent.training_model.save(f'model/model_full')
    env.close()


if __name__ == '__main__':
    main()
