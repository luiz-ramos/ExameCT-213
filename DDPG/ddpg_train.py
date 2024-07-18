import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
import tensorflow as tf
import gymnasium as gym
from utils import preprocess, decode_model_output, OUActionNoise
from networks import get_actor, get_critic
import numpy as np
import matplotlib.pyplot as plt

# Specify the `render_mode` parameter to show the attempts of the agent in a pop up window.
env = gym.make("CarRacing-v2", render_mode='human', continuous=True)

# Network parameters
state_shape = env.observation_space.shape
print("Size of State Space ->  {}".format(state_shape))
num_actions = 2
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high
lower_bound = env.action_space.low

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor("Actor", state_shape, num_actions)
critic_model = get_critic("Critic", state_shape, num_actions)

target_actor = get_actor("TargetActor", state_shape, num_actions)
target_critic = get_critic("TargetCritic", state_shape, num_actions)

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.0001

critic_optimizer = keras.optimizers.Adam(critic_lr)
actor_optimizer = keras.optimizers.Adam(actor_lr)

total_episodes = 500
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005


# Save and load weights management
def load_weights():
    actor_model.load_weights("best_solution/actor.weights.h5")
    target_actor.load_weights("best_solution/target_actor.weights.h5")
    critic_model.load_weights("best_solution/critic.weights.h5")
    target_critic.load_weights("best_solution/target_critic.weights.h5")


def save_weights():
    actor_model.save_weights("actor.weights.h5")
    target_actor.save_weights("target_actor.weights.h5")
    critic_model.save_weights("critic.weights.h5")
    target_critic.save_weights("target_critic.weights.h5")


if os.path.exists('actor.weights.h5'):
    print('Loading weights from previous learning session.')
    load_weights()
else:
    print('No weights found from previous learning session.')


# Memory Buffer for training
class Buffer:
    def __init__(self, buffer_capacity=1000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, state_shape[0], state_shape[1], state_shape[2]))
        self.action_buffer = np.zeros((self.buffer_capacity, 2))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_shape[0], state_shape[1], state_shape[2]))

    # Takes (s,a,r,s') observation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
            self,
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -keras.ops.mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = keras.ops.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = keras.ops.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = keras.ops.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = keras.ops.cast(reward_batch, dtype="float32")
        next_state_batch = keras.ops.convert_to_tensor(
            self.next_state_buffer[batch_indices]
        )

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


buffer = Buffer(1000, 64)


def policy(state, noise_object):
    noise = noise_object()

    # Get result from a network
    tensor_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
    actor_output = actor_model(tensor_state).numpy()
    actor_output = actor_output[0] + noise
    env_action = decode_model_output(actor_output)
    env_action = np.clip(np.array(env_action), a_min=env.action_space.low, a_max=env.action_space.high)
    return [env_action, actor_output]


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(target, original, tau):
    target_weights = target.get_weights()
    original_weights = original.get_weights()

    for i in range(len(target_weights)):
        target_weights[i] = original_weights[i] * tau + target_weights[i] * (1 - tau)

    target.set_weights(target_weights)


# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

for ep in range(total_episodes):
    prev_state, _ = env.reset()
    prev_state = preprocess(prev_state)

    done = False

    negative_reward_counter = 0
    time_frame_counter = 1
    episodic_reward = 0
    while True:
        action, train_action = policy(prev_state, ou_noise)
        action /= 4

        if time_frame_counter % 20 == 0:
            print(action)

        # Receive state and reward from environment.
        state, reward, done, truncated, _ = env.step(action)
        state = preprocess(state)

        # Accumulate reward
        cumulative_reward = gamma * episodic_reward + reward

        buffer.record((prev_state, train_action, reward, state))
        episodic_reward += reward

        buffer.learn()

        update_target(target_actor, actor_model, tau)
        update_target(target_critic, critic_model, tau)

        # End this episode when `done` or `truncated` is True
        if reward < 0:
            negative_reward_counter += 1
            if negative_reward_counter > 200:
                break
        else:
            no_reward_counter = 0

        prev_state = state
        time_frame_counter += 1

    ep_reward_list.append(episodic_reward)

    if ep % 20 == 0:
        save_weights()

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()
