import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import matplotlib.pyplot as plt
from DeepQLearning.dqn_agent import DQNAgent
from DeepQLearning.utils import test_nn as test_dqn_nn
import gymnasium as gym
from PPO.ppo_agent import  PPOAgent
from PPO.utils import test_NN, plot_result

RENDER = False
NUM_TEST_EPISODES = 30
fig_format = 'png'
# method = 'DQN'
# method = 'DDPG'
method = 'PPO'

if __name__ == '__main__':
    # Initiating the Mountain Car environment
    env = gym.make('CarRacing-v2', domain_randomize=False, render_mode='human', continuous=True)

    if method == 'DQN':
        original_state_size = env.observation_space.shape
        agent = DQNAgent((original_state_size[0], original_state_size[1], 3), epsilon=0)
        if os.path.exists('saves/car_racing_DQN.weights.h5'):
            print('Loading weights.')
            agent.load('saves/car_racing_DQN.weights.h5')

            total_return_history, cumulative_return_history = test_dqn_nn(agent, env, NUM_TEST_EPISODES)

            plt.plot(total_return_history, 'b')
            plt.xlabel('Episode')
            plt.ylabel('Total Return')
            plt.show(block=False)
            plt.savefig('DQN_test_total.' + fig_format)
            plt.close()

            plt.plot(cumulative_return_history, 'b')
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Return')
            plt.show(block=False)
            plt.savefig('DQN_test_cumulative.' + fig_format)
            plt.close()

        
            print('Total Rewards(mean): {:.2}, Cumulative Rewards(mean): {:.2}'.format(sum(total_return_history)/NUM_TEST_EPISODES, sum(cumulative_return_history)/NUM_TEST_EPISODES))
            
    elif method == 'DDPG':
        a = 1
    elif method == 'PPO':
        plot_result('PPO/results.csv')
        agent = PPOAgent(env)
        agent.load_model("PPO/temp")
        test_NN(agent, 1)
    
    env.close()
