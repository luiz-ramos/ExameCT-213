from actor import Actor
from critic import Critic

from collections import deque
import tensorflow
import random


def _generate_tensorflow_session():
    """
    Generates and returns the tensorflow session
    :return: the Tensorflow Session
    """
    config = tensorflow.ConfigProto()
    config.gpu_options.allow_growth = True
    return tensorflow.Session(config=config)


class DDPGAgent(object):
    """
    Represents a Deep Deterministic Policy Gradient (DDPG) agent.
    """

    def __init__(self, state_size, action_size, actor_hidden_units=(20, 40),
                 actor_learning_rate=0.0001, critic_hidden_units=(20, 40),
                 critic_learning_rate=0.001, gamma=0.95,
                 buffer_size=4098, learning_rate=0.001):
        """
        Constructs a DDPG Agent with the given parameters

        :param state_size: Int denoting the world's state dimensionality
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
        :param learning_rate:
        """

        self._discount = gamma
        self._memory_size = buffer_size

        tensorflow_session = _generate_tensorflow_session

        self._actor = Actor(tensorflow_session=tensorflow_session,
                            state_size=state_size, action_size=action_size,
                            hidden_units=actor_hidden_units,
                            learning_rate=actor_learning_rate,
                            tau=learning_rate)

        self._critic = Critic(tensorflow_session=tensorflow_session,
                              state_size=state_size, action_size=action_size,
                              hidden_units=critic_hidden_units,
                              learning_rate=critic_learning_rate,
                              tau=learning_rate)

        self._memory = deque()

    def act(self, state):
        """
        Returns the best action predicted by the agent given the current state.
        :param state: numpy array denoting the current state.
        :return: numpy array denoting the predicted action.
        """
        return self._actor.get_action(state)

    def train(self, batch_size):
        """
        Trains the DDPG Agent from its current memory

        Please note that the agent must have gone through more steps than the
        specified batch size before this method will do anything

        :return: None
        """
        if len(self._memory) > batch_size:
            self._train(batch_size)

    def _train(self, batch_size):
        """
        Helper method for train. Takes care of sampling, and training and
        updating both the actor and critic networks

        :return: None
        """
        states, actions, rewards, done, next_states = self._get_sample(batch_size)
        self._train_critic(states, actions, next_states, done, rewards)
        self._train_actor(states)
        self._update_target_models()

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

    def _train_critic(self, states, actions, next_states, done, rewards):
        """
        Trains the critic network

        C(s, a) -> q

        :param states: List of the states to train the network with
        :param actions: List of the actions to train the network with
        :param next_states: List of the t+1 states to train the network with
        :param rewards: List of rewards to calculate q_targets.

        :return: None
        """
        q_targets = self._get_q_targets(next_states, done, rewards)
        self._critic.train(states, actions, q_targets)

    def _get_q_targets(self, next_states, done, rewards):
        """
        Calculates the q targets with the following formula

        q = r + gamma * next_q

        unless there is no next state in which

        q = r

        :param next_states: List(List(Float)) Denoting the t+1 state
        :param done: List(Bool) denoting whether each step was an exit step
        :param rewards: List(Float) Denoting the reward given in each step
        :return: The q targets
        """
        next_actions = self._actor.get_target_action(next_states)
        next_q_values = self._critic.get_target_q(next_states,
                                                  next_actions)
        q_targets = [reward if this_done else reward + self._discount * next_q_value
                     for (reward, next_q_value, this_done)
                     in zip(rewards, next_q_values, done)]
        return q_targets

    def _train_actor(self, states):
        """
        Trains the actor network using the calculated deterministic policy
            gradients.

        :param states: List(List(Float)) denoting he states to train the Actor
            on
        :return: None
        """
        gradients = self._get_gradients(states)
        self._actor.train(states, gradients)

    def _get_gradients(self, states):
        """
        Calculates the Deterministic Policy Gradient for Actor training
        :param states: The states to calculate the gradients for.
        :return:
        """
        action_for_gradients = self._actor.get_action(states)
        return self._critic.get_gradients(states, action_for_gradients)

    def _update_target_models(self):
        """
        Updates the target models to slowly track the main models

        :return: None
        """
        self._critic.train_target_model()
        self._actor.train_target_model()

    def remember(self, state, action, reward, done, next_state):
        """
        Stores the given state, action, reward etc in the Agent's memory.

        :param state: The state to remember
        :param action: The action to remember
        :param reward: The reward to remember
        :param done: Whether this was a final state
        :param next_state: The next state (if applicable)
        :return: None
        """
        self._memory.append((state, action, reward, done, next_state))
        if len(self._memory) > self._memory_size:
            self._memory.popleft()
