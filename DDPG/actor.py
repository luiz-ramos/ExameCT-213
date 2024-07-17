import tensorflow
from tensorflow.keras import models, layers, optimizers, backend, activations, losses
import keras
from keras import initializers
import numpy as np


class Actor(object):
    """
    Object representing the actor network, which approximates the function:

        u(s) -> a

    where u (actually mew) is the deterministic policy mapping from states s to
    actions a.
    """

    def __init__(self, state_shape, action_size,
                 hidden_units, learning_rate=0.0001,
                 tau=0.001):
        """
        Constructor for the Actor network

        :param state_shape: An array denoting the dimensionality of the states
            in the current problem
        :param action_size: An integer denoting the dimensionality of the
            actions in the current problem
        :param hidden_units: An iterable defining the number of hidden units in
            each layer. Soon to be depreciated. default: (20, 40)
        :param learning_rate: A fload denoting the speed at which the network
            will learn. default: 0.0001
        :param tau: A flot denoting the rate at which the target model will
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
        self._states = None
        self._state_shape = state_shape
        self._action_size = action_size
        self._tau = tau
        self._learning_rate = learning_rate
        self._hidden = hidden_units

        # Generate the main model
        self._model, self._model_weights, self._model_input = \
            self._generate_model()
        # Generate carbon copy of the model so that we avoid divergence
        self._target_model, self._target_weights, self._target_state = \
            self._generate_model()

    def policy(self, state):
        """
        Returns the best action predicted by the target model agent given the current state.
        :param state: numpy array denoting the list of states to be used.
        :return: numpy array denoting the predicted actions for each state.
        """
        sampled = keras.ops.squeeze(self._model(state))
        sampled = sampled.numpy()
        return sampled

    def get_action(self, states):
        """
        Returns the best action predicted by the target model agent given the current state.
        :param states: numpy array denoting the list of states to be used.
        :return: numpy array denoting the predicted actions for each state.
        """
        return self._model(states, training=True)

    def get_target_actions(self, states):
        """
        Returns the best action predicted by the target model agent given the current state.
        :param states: numpy array denoting the list of states to be used.
        :return: numpy array denoting the predicted actions for each state.
        """
        return self._target_model(states, training=True)

    def get_trainable_params(self):
        return self._model.trainable_variables

    def train(self, states, action_gradients):
        """
        Updates the weights of the main network

        :param states: The states of the input to the network
        :param action_gradients: The gradients of the actions to update the
            network
        :return: None
        """
        action_gradients = action_gradients[0]
        self._adam_optimizer([states, action_gradients])

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

        :return: at tuple containing references to the model, weights,
            and input later
        """
        input_layer = layers.Input(shape=self._state_shape)

        "Convolution to deal with 2D-Input"
        conv1layer = layers.Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu')(input_layer)
        pool1 = layers.MaxPool2D(pool_size=(2, 2))(conv1layer)
        conv2layer = layers.Conv2D(filters=12, kernel_size=(4, 4), strides=1, activation='relu')(pool1)
        pool2 = layers.MaxPool2D(pool_size=(2, 2))(conv2layer)
        flatten = layers.Flatten()(pool2)

        layer = layers.Dense(self._hidden[0], activation='relu')(flatten)
        layer = layers.Dense(self._hidden[1], activation='relu')(layer)

        steering = layers.Dense(1, activation='tanh',
                                kernel_initializer=initializers.RandomUniform(minval=-0.5, maxval=0.5))(layer)
        brake = layers.Dense(1, activation='sigmoid',
                             kernel_initializer=initializers.RandomUniform(minval=0.25, maxval=0.75))(layer)
        gas = layers.Dense(1, activation='sigmoid',
                           kernel_initializer=initializers.RandomUniform(minval=0.25, maxval=0.75))(layer)
        output_layer = layers.Concatenate()([steering, gas, brake])
        model = models.Model(inputs=input_layer, outputs=output_layer)

        print(model.summary())
        return model, model.trainable_weights, input_layer

    def load(self):
        """
        Loads the neural network's weights from disk.
        """
        self._model.load_weights("actor.h5")
        self._target_model.load_weights("target_actor.h5")

    def save(self):
        """
        Saves the neural network's weights to disk.
        """
        self._model.save_weights("actor.h5")
        self._target_model.save_weights("target_actor.h5")
