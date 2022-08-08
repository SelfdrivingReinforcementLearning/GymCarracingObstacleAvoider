import random
from Agent import Agent
from gym.envs.box2d.car_racing import CarRacing


def main():
    episodes = 2000
    env = CarRacing()
    agent = Agent()
    update_steps = 0

    for i in range(1, episodes+1):
        print(f'Episode {i}')
        state = env.reset()
        done = False
        episode_reward = 0
        negative_reward_streak = 0

        while not done:
            print(update_steps)
            action = agent.get_action(state)
            print(episode_reward)
            old_state = state
            state, reward, done, truncated, _ = env.step(action)
            if reward < 0:
                negative_reward_streak += 1
            else:
                negative_reward_streak = 0
            episode_reward += reward
            agent.buffer.append((old_state, state, action, done, reward, truncated))
            if len(agent.buffer) >= agent.BATCH_SIZE:
                sample = random.sample(agent.buffer, agent.BATCH_SIZE)
                agent.train(sample)
                update_steps += 1
                if update_steps == 5:
                    agent.target_model.set_weights(agent.training_model.get_weights())
                    update_steps = 0
            if truncated or episode_reward < -50 or negative_reward_streak >= 50:
                break

        agent.epsilon = agent.epsilon * agent.DECAY_RATE
        if agent.epsilon < agent.MIN_EPSILON:
            agent.epsilon = agent.MIN_EPSILON
        if i % 100 == 0:
            agent.training_model.save(f'model/model_episodes{i}')

    agent.training_model.save(f'model/model_full')
    env.close()


if __name__ == '__main__':
    main()
