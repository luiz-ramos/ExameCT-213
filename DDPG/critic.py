import tensorflow
from tensorflow.keras import models, layers, optimizers, backend, activations, losses
import keras
import numpy as np


class Critic(object):
    def __init__(self, state_shape, action_size,
                 hidden_units, learning_rate=0.0001,
                 tau=0.001):
        """
        Constructor for the Critic network

        :param state_shape: An array denoting the dimensionality of the states
            in the current problem
        :param action_size: An integer denoting the dimensionality of the
            actions in the current problem
        :param hidden_units: An iterable defining the number of hidden units in
            each layer. Soon to be depreciated. default: (20, 40)
        :param learning_rate: A fload denoting the speed at which the network
            will learn. default: 0.0001
        :param tau: A float denoting the rate at which the target model will
            track the main model. Formally, the tracking function is defined as:

              target_weights = tau * main_weights + (1 - tau) * target_weights

            for more explanation on how and why this happens,
            please refer to the DDPG paper:

            Lillicrap, Hunt, Pritzel, Heess, Erez, Tassa, Silver, & Wiestra.
            Continuous Control with Deep Reinforcement Learning. arXiv preprint
            arXiv:1509.02971, 2015.

            default: 0.001
        """
        # Store parameters
        self._state_inputs = None
        self._state_shape = state_shape
        self._action_size = action_size
        self._tau = tau
        self._learning_rate = learning_rate
        self._hidden = hidden_units

        # Generate the main model
        self._model, self._state_input, self._action_input = \
            self._generate_model()
        # Generate carbon copy of the model so that we avoid divergence
        self._target_model, self._target_weights, self._target_state = \
            self._generate_model()

    def get_q(self, states, actions):
        """
        Returns the best action predicted by the target model agent given the current state.
        :param states: numpy array denoting the list of states to be used.
        :param actions: numpy array denoting the list of predicted actions to be used.
        :return: numpy array denoting the predicted actions.
        """
        return self._model([states, actions], training=True)

    def get_target_q(self, states, actions):
        """
        Returns the best action predicted by the target model agent given the current state.
        :param states: numpy array denoting the list of states to be used.
        :param actions: numpy array denoting the list of predicted actions to be used.
        :return: numpy array denoting the predicted actions.
        """
        return self._target_model([states, actions], training=True)

    def get_trainable_params(self):
        return self._model.trainable_variables

    def train_target_model(self):
        """
        Updates the weights of the target network to slowly track the main
        network.

        The speed at which the target network tracks the main network is
        defined by tau, given in the constructor to this class. Formally,
        the tracking function is defined as:

            target_weights = tau * main_weights + (1 - tau) * target_weights

        :return: None
        """
        main_weights = self._model.get_weights()
        target_weights = self._target_model.get_weights()
        target_weights = [self._tau * main_weight + (1 - self._tau) *
                          target_weight for main_weight, target_weight in
                          zip(main_weights, target_weights)]
        self._target_model.set_weights(target_weights)

    def _generate_model(self):
        """
        Generates the model based on the hyperparameters defined in the
        constructor.

        :return: at tuple containing references to the model, state input layer,
            and action input later
        """
        state_input_layer = layers.Input(shape=self._state_shape)
        action_input_layer = layers.Input(shape=[self._action_size])

        "Convolution to deal with 2D-Input"
        conv1layer = layers.Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu')(state_input_layer)
        pool1 = layers.MaxPool2D(pool_size=(2, 2))(conv1layer)
        conv2layer = layers.Conv2D(filters=12, kernel_size=(4, 4), strides=1, activation='relu')(pool1)
        pool2 = layers.MaxPool2D(pool_size=(2, 2))(conv2layer)
        flatten = layers.Flatten()(pool2)

        s_layer = layers.Dense(self._hidden[0], activation='relu')(flatten)
        a_layer = layers.Dense(self._hidden[1], activation='linear')(action_input_layer)

        hidden = layers.Dense(self._hidden[1], activation='linear')(s_layer)
        hidden = layers.Concatenate()([hidden, a_layer])
        hidden = layers.Dense(self._hidden[1], activation='relu')(hidden)
        output_layer = layers.Dense(1, activation='linear')(hidden)
        model = models.Model(inputs=[state_input_layer, action_input_layer],
                             outputs=output_layer)

        print(model.summary())
        return model, state_input_layer, action_input_layer

    def load(self):
        """
        Loads the neural network's weights from disk.
        """
        self._model.load_weights("critic.h5")
        self._target_model.load_weights("target_critic.h5")

    def save(self):
        """
        Saves the neural network's weights to disk.
        """
        self._model.save_weights("critic.h5")
        self._target_model.save_weights("target_critic.h5")
