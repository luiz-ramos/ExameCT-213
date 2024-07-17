from DDPG.actor import Actor
from DDPG.critic import Critic

from collections import deque
import tensorflow
import keras
import random
import numpy as np


def _generate_tensorflow_session():
    """
    Generates and returns the tensorflow session
    :return: the Tensorflow Session
    """
    config = tensorflow.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tensorflow.compat.v1.Session(config=config)


class DDPGAgent(object):
    """
    Represents a Deep Deterministic Policy Gradient (DDPG) agent.
    """

    def __init__(self, state_shape, action_size, actor_hidden_units=(20, 40),
                 actor_learning_rate=0.0001, critic_hidden_units=(20, 40),
                 critic_learning_rate=0.001, gamma=0.95,
                 buffer_size=4098, tau=0.01):
        """
        Constructs a DDPG Agent with the given parameters

        :param state_shape: Int denoting the world's state dimensionality
        :param action_size: Int denoting the world's action dimensionality
        :param actor_hidden_units: Tuple(Int) denoting the actor's hidden layer
            sizes. Each element in the tuple represents a layer in the Actor
            network and the Int denotes the number of neurons in the layer.
        :param actor_learning_rate: Float denoting the learning rate of the
            Actor network. Best to be some small number close to 0.
        :param critic_hidden_units: Tuple(Int) denoting the critic's hidden
            layer sizes. Each element in the tuple represents a layer in the
            Critic network and the Int denotes the number of neurons in the
            layer.
        :param critic_learning_rate: Float denoting the learning rate of the
            Critic network. Best to be some small number close to 0.
        :param gamma: Float denoting the discount (gamma) given to future
            potential rewards when calculating q values
        :param buffer_size: Int denoting the number of State, action, rewards
            that the agent will remember
        :param tau: A float denoting the rate at which the target model will
            track the main model.
        """

        self._gamma = gamma
        self._buffer_size = buffer_size

        self._actor = Actor(state_shape=state_shape, action_size=action_size,
                            hidden_units=actor_hidden_units,
                            learning_rate=actor_learning_rate,
                            tau=tau)

        self._actor_optimizer = keras.optimizers.Adam(actor_learning_rate)

        self._critic = Critic(state_shape=state_shape, action_size=action_size,
                              hidden_units=critic_hidden_units,
                              learning_rate=critic_learning_rate,
                              tau=tau)

        self._critic_optimizer = keras.optimizers.Adam(critic_learning_rate)

        self._memory = deque()

    def act(self, state):
        """
        Returns the best action predicted by the agent given the current state.
        :param state: numpy array denoting the current state.
        :return: numpy array denoting the predicted action.
        """
        action = self._actor.policy(state)
        action = np.squeeze(action)
        return [action]

    def train(self, batch_size):
        """
        Trains the DDPG Agent from its current memory

        Please note that the agent must have gone through more steps than the
        specified batch size before this method will do anything

        :return: None
        """
        if len(self._memory) > 2 * batch_size:
            self._train(batch_size)

    def _train(self, batch_size):
        """
        Helper method for train. Takes care of sampling, and training and
        updating both the actor and critic networks

        :return: None
        """
        states, actions, rewards, done, next_states = self._get_sample(batch_size)

        # Transforming into tensor
        states = keras.ops.convert_to_tensor(np.array(states))
        next_states = keras.ops.convert_to_tensor(np.array(next_states))
        rewards = keras.ops.convert_to_tensor(np.array(rewards))
        rewards = keras.ops.cast(rewards, dtype="float32")
        actions = keras.ops.convert_to_tensor(np.array(actions))

        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tensorflow.GradientTape() as tape:
            target_actions = self._actor.get_target_actions(next_states)
            y = rewards + self._gamma * self._critic.get_target_q(states, target_actions)
            critic_value = self._critic.get_q(states, actions)
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self._critic.get_trainable_params())
        self._critic_optimizer.apply_gradients(
            zip(critic_grad, self._critic.get_trainable_params())
        )

        with tensorflow.GradientTape() as tape:
            actions_pred = self._actor.get_action(states)
            critic_value = self._critic.get_q(states, actions_pred)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -keras.ops.mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self._actor.get_trainable_params())
        self._actor_optimizer.apply_gradients(
            zip(actor_grad, self._actor.get_trainable_params())
        )

        print("Losses:")
        print(critic_loss.numpy())
        print(actor_loss.numpy())

        self._actor.train_target_model()
        self._critic.train_target_model()

    def _get_sample(self, batch_size):
        """
        Finds a random sample of size self._batch_size from the agent's current
        memory.

        :return: Tuple(List(Float, Boolean)) denoting the sample of states,
            actions, rewards, done, and next states.
        """
        sample = random.sample(self._memory, batch_size)
        states, actions, rewards, done, next_states = zip(*sample)
        return states, actions, rewards, done, next_states

    def remember(self, state, action, reward, done, next_state):
        """
        Stores the given state, action, reward etc. in the Agent's memory.

        :param state: The state to remember
        :param action: The action to remember
        :param reward: The reward to remember
        :param done: Whether this was a final state
        :param next_state: The next state (if applicable)
        :return: None
        """
        if len(self._memory) + 1 > self._buffer_size:
            self._memory.popleft()
        self._memory.append((state, action, reward, done, next_state))

    def load(self):
        """
        Loads the neural network's weights from disk.
        """
        self._actor.load()
        self._critic.load()

    def save(self):
        """
        Saves the neural network's weights to disk.
        """
        self._actor.save()
        self._critic.save()
