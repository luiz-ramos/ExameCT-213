import numpy as np
import tensorflow as tf
from keras import models, layers, activations, losses, Input, Model, optimizers


class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0] + 1
        self.policy, self.critic = self._build_model()
        self.actor_lr = 0.0003
        self.critic_lr = 0.001
        self.gamma = 0.96
        self.eps_clip = 0.2
        self.policy, self.critic = self._build_model()
        self.actor_optimizer = optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = optimizers.Adam(learning_rate=self.critic_lr)
        self.lam = 1

    def _build_model(self):
        inputs = Input(shape=(96, 96, 3))

        x = layers.Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation=activations.relu)(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(filters=12, kernel_size=(4, 4), strides=(1, 1), activation=activations.relu)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=216, activation=activations.relu)(x)

        # output 1: action_mean
        action_mean = layers.Dense(self.action_dim, activation='sigmoid')(x)
        # output 2: critic
        critic = layers.Dense(1)(x)

        actor = Model(inputs=inputs, outputs=action_mean)
        critic = Model(inputs=inputs, outputs=critic)

        return actor, critic

    def save_model(self, filename):
        self.policy.save_weights(f"{filename}_actor.weights.h5")
        self.critic.save_weights(f"{filename}_critic.weights.h5")

    def load_model(self, filename):
        self.policy.load_weights(f"{filename}_actor.weights.h5")
        self.critic.load_weights(f"{filename}_critic.weights.h5")

    def discount_rewards(self, rewards):
        returns = []
        discounted_sum = 0.0
        for reward in reversed(rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.append(discounted_sum)
        returns.reverse()
        return tf.convert_to_tensor(returns, dtype=tf.float32)

    def ppo_loss(self, new_actions, old_action_means, actions, discounted_rewards, values, eps_clip):
        std = 0.2
        log_probs = -0.5 * tf.reduce_sum(((new_actions - actions) / std) ** 2 + 2 * np.log(std) + np.log(2 * np.pi),
                                         axis=1)
        old_log_probs = -0.5 * tf.reduce_sum(
            ((old_action_means - actions) / std) ** 2 + 2 * np.log(std) + np.log(2 * np.pi), axis=1)

        ratios = tf.exp(log_probs - old_log_probs)
        advantages = discounted_rewards - values
        surr1 = ratios * advantages
        surr2 = tf.clip_by_value(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

        return -tf.reduce_mean(tf.minimum(surr1, surr2))

    def optimize_policy(self, states, actions, rewards, old_action_means):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        old_action_means = tf.convert_to_tensor(old_action_means, dtype=tf.float32)

        discounted_rewards = self.discount_rewards(rewards)
        discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)

        # Compile the policy model
        self.policy.compile(optimizer=self.actor_optimizer,
                             loss=lambda y_true, y_pred: self.ppo_loss(y_pred, old_action_means, actions, discounted_rewards,
                                                                  self.critic(states), self.eps_clip))

        # Compile the critic model
        self.critic.compile(optimizer=self.critic_optimizer, loss='mse')

        for _ in range(10):
            # Train the policy (actor) model
            self.policy.train_on_batch(states, actions)

            # Train the critic model
            self.critic.train_on_batch(states, discounted_rewards)
