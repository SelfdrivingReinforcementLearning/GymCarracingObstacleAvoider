import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from Agent import Agent
from gym.envs.box2d.car_racing import CarRacing


def main():
    episodes = 20000
    env = CarRacing()
    agent = Agent()
    loss_values = []
    accuracy_values = []
    reward_values = []
    episode_list = []
    hist = None
    model_name = 'Model_Eps0.999'

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
                hist = agent.train(sample)
                update_steps += 1
                if update_steps == 15:
                    agent.target_model.set_weights(agent.training_model.get_weights())
                    update_steps = 0
            if truncated or episode_reward < -50 or negative_reward_streak >= 25:
                done = True

            if done:
                if hist:
                    loss = hist.history['loss'][0]
                    accuracy = hist.history['accuracy'][0]
                else:
                    loss = -1.0
                    accuracy = -1.0
                loss_values.append(loss)
                accuracy_values.append(accuracy)
                episode_list.append(episode)
                reward_values.append(episode_reward)

        agent.epsilon = agent.epsilon * agent.DECAY_RATE
        if agent.epsilon < agent.MIN_EPSILON:
            agent.epsilon = agent.MIN_EPSILON
        if episode % 25 == 0:
            agent.training_model.save(f'models/{model_name}{episode}')
            plt.plot(np.array(episode_list), np.array(loss_values), label='Loss')
            plt.plot(np.array(episode_list), np.array(accuracy_values), label='Accuracy')
            plt.xlabel('Episodes')
            plt.ylabel('Loss and Accuracy')
            plt.title('Loss and Accuracy by Episode')
            plt.legend()
            plt.savefig(f'logs/{model_name}_loss_and_accuracy.png', format='png')

            plt.figure()
            plt.plot(np.array(episode_list), np.array(reward_values), label='Episode Reward')
            plt.xlabel('Episodes')
            plt.ylabel('Episode Reward')
            plt.title('Total Episode Reward by Episode')
            plt.legend()
            plt.savefig(f'logs/{model_name}_episode_reward.png', format='png')

            plt.show()

    agent.training_model.save(f'model/model_full')
    env.close()


if __name__ == '__main__':
    main()
