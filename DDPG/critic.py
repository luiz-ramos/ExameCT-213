import tensorflow
from tensorflow.keras import models, layers, optimizers, backend, activations, losses
import numpy as np


class Critic(object):
    def __init__(self, tensorflow_session, state_shape, action_size,
                 hidden_units, learning_rate=0.0001,
                 tau=0.001):
        """
        Constructor for the Actor network

        :param tensorflow_session: The tensorflow session.
            See https://www.tensorflow.org for more information on tensorflow
            sessions.
        :param state_shape: An array denoting the dimensionality of the states
            in the current problem
        :param action_size: An integer denoting the dimensionality of the
            actions in the current problem
        :param hidden_units: An iterable defining the number of hidden units in
            each layer. Soon to be depreciated. default: (300, 600)
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
        self._tensorflow_session = tensorflow_session
        self._state_shape = state_shape
        self._action_size = action_size
        self._tau = tau
        self._learning_rate = learning_rate
        self._hidden = hidden_units

        # Let tensorflow and keras work together
        tensorflow.compat.v1.keras.backend.set_session(tensorflow_session)

        # Generate the main model
        self._model, self._state_input, self._action_input = \
            self._generate_model()
        # Generate carbon copy of the model so that we avoid divergence
        self._target_model, self._target_weights, self._target_state = \
            self._generate_model()

        # Gradients for policy update
        self._action_gradients = tensorflow.compat.v1.gradients(self._model.output,
                                                                self._action_input)
        print(type(self._action_gradients))

        tensorflow.compat.v1.global_variables_initializer()

    def get_gradients(self, states, actions):
        """
        Returns the gradients.
        :param states:
        :param actions:
        :return:
        """
        return self._tensorflow_session.run(self._action_gradients, feed_dict={
            self._state_inputs: states,
            self._action_input: actions
        })[0]

    def get_q(self, state, action):
        """
        Returns the best action predicted by the agent given the current state.
        :param state: numpy array denoting the current state.
        :return: numpy array denoting the predicted action.
        :action: numpy array denoting the predicted action.
        """
        return self._model.predict(state, action)

    def get_target_q(self, states, actions):
        """
        Returns the best action predicted by the target model agent given the current state.
        :param states: numpy array denoting the list of states to be used.
        :param actions: numpy array denoting the list of predicted actions to be used.
        :return: numpy array denoting the predicted actions.
        """
        return self._target_model.predict(x=[states, actions])

    def train(self, states, actions, q_targets):
        """
        Trains the critic network.

        :param states:
        :param actions:
        :param q_targets:
        """
        self._model.train_on_batch(x=[states, actions], sample_weight=q_targets)

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
        conv1layer = layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation='relu')(state_input_layer)
        conv2layer = layers.Conv2D(filters=64, kernel_size=(8, 8), strides=(4, 4), activation='relu')(conv1layer)
        flatten = layers.Flatten()(conv2layer)

        s_layer = layers.Dense(self._hidden[0], activation='relu')(flatten)
        a_layer = layers.Dense(self._hidden[1], activation='linear')(action_input_layer)

        hidden = layers.Dense(self._hidden[1], activation='linear')(s_layer)
        hidden = layers.Add()([hidden, a_layer])
        hidden = layers.Dense(self._hidden[1], activation='relu')(hidden)
        output_layer = layers.Dense(self._action_size, activation='linear')(hidden)
        model = models.Model(inputs=[state_input_layer, action_input_layer],
                             outputs=output_layer)
        model.compile(loss='mse', optimizer=optimizers.legacy.Adam(lr=self._learning_rate))
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
