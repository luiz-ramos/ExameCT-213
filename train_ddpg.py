import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from utils import preprocess_state
import tensorflow as tf
from DDPG.ddpg_agent import DDPGAgent

NUM_EPISODES = 3000  # Number of episodes used for training
RENDER = False  # If the Mountain Car environment should be rendered
fig_format = 'png'  # Format used for saving matplotlib's figures
# fig_format = 'eps'
# fig_format = 'svg'

# Comment this line to enable training using your GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.compat.v1.disable_eager_execution()

# Initiating the Car Racing environment
env = gym.make("CarRacing-v2", render_mode='human')
state_shape = env.observation_space.shape
state_shape = (state_shape[0], state_shape[1], 1)
action_size = env.action_space.shape[0]

# Creating the DDPG agent
agent = DDPGAgent(state_shape, action_size)

# Checking if weights from previous learning session exists
if os.path.exists('actor.h5'):
    print('Loading weights from previous learning session.')
    agent.load()
else:
    print('No weights found from previous learning session.')
done = False
batch_size = 64  # batch size used for the experience replay
return_history = []

for episodes in range(1, NUM_EPISODES + 1):
    # Reset the environment
    ob = env.reset()

    # Isolating the matrix
    ob = ob[0]

    # Extracting state from observed state
    state = preprocess_state(ob)

    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = 0.0
    while not done:
        if RENDER:
            env.render()  # Render the environment for visualization

        # Select action
        action = agent.act(state)[0]

        # Take action, observe reward and new state
        ob, reward, done, fa, info = env.step(action)
        next_state = preprocess_state(ob)

        # Making reward engineering to allow faster training
        # reward = utils.reward_engineering(state[0], action, reward, next_state[0], done)

        # Appending this experience to the experience replay buffer
        agent.remember(state, action, reward, done, next_state)

        # Accumulate reward
        cumulative_reward = agent._gamma * cumulative_reward + reward
        if done:
            print("episode: {}/{}, score: {:.6}"
                  .format(episodes, NUM_EPISODES, cumulative_reward))
            break

        # Update state
        state = next_state

        # We only update the policy if we already have enough experience in memory
        loss = agent.train(batch_size)
    return_history.append(cumulative_reward)
    # Every 10 episodes, update the plot for training monitoring
    if episodes % 20 == 0:
        plt.plot(return_history, 'b')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.show(block=False)
        plt.pause(0.1)
        plt.savefig('dqn_training.' + fig_format)
        # Saving the model to disk
        agent.save()
plt.pause(1.0)
