import numpy as np
import gymnasium as gym
import tensorflow as tf
import keras
from utils import preprocess, decode_model_output

def test_ddpg(env):
    actor = keras.models.load_model('best_solution/actor.weights.h5')
    critic = keras.models.load_model('best_solution/critic.weights.h5')
    target_actor = keras.models.load_model('best_solution/target_actor.weights.h5')
    target_critic = keras.models.load_model('best_solution/target_critic.weights.h5')

    def policy(state):
        # Get result from a network
        tensor_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        actor_output = actor(tensor_state).numpy()
        actor_output = actor_output[0]
        env_action = decode_model_output(actor_output)
        env_action = np.clip(np.array(env_action), a_min=env.action_space.low, a_max=env.action_space.high)
        return env_action

    prev_state, _ = env.reset()
    prev_state = preprocess(prev_state)
    done = False

    while not done:
        action = policy(prev_state)
        action /= 4

        # Receive state and reward from environment.
        state, reward, done, truncated, _ = env.step(action)
        state = preprocess(state)

        prev_state = state
