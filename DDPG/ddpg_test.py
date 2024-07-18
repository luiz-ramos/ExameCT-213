import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from DDPG.utils import preprocess, decode_model_output

env = gym.make('CarRacing-v2', continuous=True, render_mode='human')

def test_ddpg(env, num_episodes=100):
    actor = keras.models.load_model('DDPG/best_solution/actor.weights.h5')
    critic = keras.models.load_model('DDPG/best_solution/critic.weights.h5')
    target_actor = keras.models.load_model('DDPG/best_solution/target_actor.weights.h5')
    target_critic = keras.models.load_model('DDPG/best_solution/target_critic.weights.h5')

    reward_history = []
    avg_reward_history = []

    def policy(state):
        # Get result from a network
        tensor_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        actor_output = actor(tensor_state).numpy()
        actor_output = actor_output[0]
        env_action = decode_model_output(actor_output)
        env_action = np.clip(np.array(env_action), a_min=env.action_space.low, a_max=env.action_space.high)
        return env_action

    for ep in range(num_episodes):
        prev_state, _ = env.reset()
        prev_state = preprocess(prev_state)
        negative_reward_counter = 0
        time_frame_counter = 1
        episodic_reward = 0
        while True:
            action = policy(prev_state)
            action /= 4

            # Receive state and reward from environment.
            state, reward, done, truncated, _ = env.step(action)
            state = preprocess(state)

            episodic_reward += reward

            # End this episode when `done` or `truncated` is True
            if reward < 0:
                negative_reward_counter += 1
                if negative_reward_counter > 200:
                    break
            else:
                negative_reward_counter = 0

            prev_state = state
            time_frame_counter += 1

        reward_history.append(episodic_reward)
        avg_reward_history.append(np.mean(reward_history[-100:]))

    # Plotting graph
    fig = plt.figure()
    ax = plt.subplot(111)
    episodes = np.arange(1, num_episodes + 1)
    ax.plot(episodes, reward_history, label="Rewards (Original)")
    ax.plot(episodes, avg_reward_history, label="Rewards (100 Moving Averages)")
    ax.legend()
    plt.show()

