import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import gym
import matplotlib.pyplot as plt
from collections import deque
from dqn_agent import DQNAgent
from utils import preprocess_state, process_deque, reward_engineering_car_racing

RENDER = False
NUM_EPISODES = 1000
STACK_FRAMES = 3
SKIP_FRAMES = 3
TRAINING_BATCH_SIZE = 64
SAVE_TRAINING_FREQUENCY = 50
UPDATE_MODEL_FREQUENCY = 5
fig_format = 'png'

if __name__ == '__main__':
    # Initiating the Mountain Car environment
    env = gym.make('CarRacing-v2', domain_randomize=False, render_mode='human', continuous=True)
    
    original_state_size = env.observation_space.shape
    # Creating the DQN agent
    agent = DQNAgent((original_state_size[0], original_state_size[1], STACK_FRAMES))
    
    # Checking if weights from previous learning session exists
    if os.path.exists('car_racing.h5'):
        print('Loading weights from previous learning session.')
        agent.load("car_racing.h5")
    else:
        print('No weights found from previous learning session.')
    done = False
    total_return_history, cumulative_return_history = [], []

    for episodes in range(1, NUM_EPISODES+1):
        # Reset the environment
        state, _ = env.reset()
        state = preprocess_state(state)
        state_stack = deque([state]*STACK_FRAMES, maxlen=STACK_FRAMES)
        
        total_reward = 0.0
        cumulative_reward = 0.0
        early_stop_counter = 0
        done = False
        
        for time in range(1, 300):
            if RENDER:
                env.render()

            curr_state_stack = process_deque(state_stack)
            action = agent.act(curr_state_stack)
            action_idx = agent.actions.index(action)

            reward = 0
            for _ in range(SKIP_FRAMES+1):
                next_state, r, terminated, truncated, _ = env.step(action)
                reward += r
                done = terminated or truncated
                if done:
                    break

            # Early Stop strategy
            if time > 100 and reward < 0:
                early_stop_counter += 1

            # Extra bonus for the model if it uses full gas
            reward = reward_engineering_car_racing(action, reward)

            total_reward += reward
            cumulative_reward = agent.gamma * cumulative_reward + reward

            next_state = preprocess_state(next_state)
            state_stack.append(next_state)
            next_state_stack = process_deque(state_stack)

            agent.append_experience(curr_state_stack, action, reward, next_state_stack, done)

            if done or early_stop_counter >= 25 or total_reward < 0:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards(adjusted): {:.2}, Cumulative Rewards: {:.2}, Epsilon: {:.2}'.format(episodes, NUM_EPISODES, time, float(total_reward), float(cumulative_reward), float(agent.epsilon)))
                total_return_history.append(total_reward)
                cumulative_return_history.append(cumulative_reward)
                break
            if len(agent.replay_buffer) > TRAINING_BATCH_SIZE:
                agent.replay(TRAINING_BATCH_SIZE)
        
        if episodes % UPDATE_MODEL_FREQUENCY == 0:
            agent.update_target_model()

        if episodes % SAVE_TRAINING_FREQUENCY == 0:
            plt.plot(total_return_history, 'b')
            plt.xlabel('Episode')
            plt.ylabel('Total Return')
            plt.show(block=False)
            plt.savefig('dqn_training_total.' + fig_format)
            plt.close()

            plt.plot(cumulative_return_history, 'b')
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Return')
            plt.show(block=False)
            plt.savefig('dqn_training_cumulative.' + fig_format)
            plt.close()

            agent.save('./save/episode_{}.weights.h5'.format(episodes))

    env.close()
