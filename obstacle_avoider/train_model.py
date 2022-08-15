import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from Agent import Agent
from gym.envs.box2d.car_racing import CarRacing


def main():
    """Runs the training loop
    """
    episodes = 20000
    env = CarRacing()
    agent = Agent()
    loss_values = []
    accuracy_values = []
    reward_values = []
    episode_list = []
    hist = None
    model_name = 'Model_Eps0.995_bigger'

    # determines how many of the frames will be used
    # e.g. if skip_frames is 4, then every 4th frame will be used
    skip_frames = 4

    # looping over all episodes
    for episode in range(1, episodes + 1):
        print(f'Episode {episode}')

        # getting the initial state, processing it and putting it four times into a deque (state_queue)
        # state_queue always keeps the last four states so that they can be fed to the agent together
        state = env.reset()
        state = cv.cvtColor(state, cv.COLOR_BGR2GRAY)
        state = state / 255.0
        state_list = [state]
        state_queue = deque(state_list * 4, maxlen=4)

        done = False
        truncated = False
        update_steps = 0
        episode_reward = 0
        negative_reward_streak = 0

        # looping through episode
        while not done:
            # enabling rendering to be able to watch the training
            env.render()

            # preprocessing the already existing states (moving axis from (4, x, y) to (x, y, 4))
            old_states_array = np.array(state_queue)
            old_states = np.moveaxis(old_states_array, 0, -1)

            # getting the action that the agent shall perform based on the already existing states
            action = agent.get_action(old_states)

            # skipping over a number of frames determined by skip_frames to combine multiple frames to one state,
            # accumulating the combined reward
            # this is done because there is not a lot of change from one state to the next one
            reward = 0
            skip_count = 0
            while not (done or truncated):
                new_state, frame_reward, done, truncated, _ = env.step(action)
                reward += frame_reward
                skip_count += 1
                if skip_count == skip_frames:
                    break

            # processing the new state, putting the new state into the state_queue and preprocessing the new state
            # (moving axis from (4, x, y) to (x, y, 4))
            new_state = cv.cvtColor(new_state, cv.COLOR_BGR2GRAY)
            new_state = new_state / 255.0
            state_queue.append(new_state)
            new_states_array = np.array(state_queue)
            next_states = np.moveaxis(new_states_array, 0, -1)

            # keeping track of how many times in a row the reward was negative
            if reward < 0:
                negative_reward_streak += 1
            else:
                negative_reward_streak = 0
            episode_reward += reward

            # writing the last experience into a buffer so that it later can be sampled and used by the agent
            agent.buffer.append((old_states, next_states, agent.actions.index(action), done, reward, truncated))

            # do training by sampling from the buffer if the buffer is big enough
            if len(agent.buffer) >= agent.BATCH_SIZE:
                sample = random.sample(agent.buffer, agent.BATCH_SIZE)
                hist = agent.train(sample)

                # update the target network every update_steps steps
                update_steps += 1
                if update_steps == 15:
                    agent.target_model.set_weights(agent.training_model.get_weights())
                    update_steps = 0

            # cancel the episode if the agent went too far away from the road, the reward over the whole episode gets
            # too small or if the agent received a negative reward too many times in a row
            if truncated or episode_reward < -50 or negative_reward_streak >= 25:
                done = True

            # save loss and accuracy at the end of every episode
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

        # decay epsilon
        agent.epsilon = agent.epsilon * agent.DECAY_RATE
        if agent.epsilon < agent.MIN_EPSILON:
            agent.epsilon = agent.MIN_EPSILON

        # save the model and log loss and accuracy per episode every 25 episodes
        if episode % 25 == 0:
            agent.training_model.save(f'models/{model_name}_{episode}')
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

    # save the complete model once training is done
    agent.training_model.save(f'models/{model_name}_full')
    env.close()


if __name__ == '__main__':
    main()
