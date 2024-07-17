import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    # Creates a Deep Q-Networks agent
    def __init__(
        self,
        state_size,
        actions = [
            (0, 0, 0), (0, 1, 0), (0, 0, 0.2), (1, 1, 0), (-1, 1, 0),
            (1, 0, 0.2), (-1, 0, 0.2), (0.5, 0.5, 0), (-0.5, 0.5, 0), 
            (0.5, 0, 0.1), (-0.5, 0, 0.1), (1, 0, 0), (-1, 0, 0)
        ],
        buffer_size     = 5000,
        gamma           = 0.95,  # discount rate
        epsilon         = 1.0,   # exploration rate
        epsilon_min     = 0.1,
        epsilon_decay   = 0.9999,
        learning_rate   = 0.001
    ):
        self.state_size = state_size
        self.actions = actions
        self.replay_buffer = deque(maxlen=buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self.make_model()
        self.target_model = self.make_model()
        self.update_target_model()

    def make_model(self):
        # Makes the action-value neural network model using Keras
        model = Sequential()
        # 0 layer
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=self.state_size))
        # 1 layer
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # 2 layer
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        # 3 layer
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # 4 layer
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        # 5 layer
        model.add(Dense(len(self.actions), activation=None))
        
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-7))
        
        model.summary()
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        # Chooses an action using an epsilon-greedy policy
        action_idx = 0
        if np.random.rand() > self.epsilon:
            q_state = self.model.predict(state, verbose=0)
            action_idx = np.argmax(q_state[0])
        else:
            action_idx = random.randrange(len(self.actions))
        return self.actions[action_idx]

    def append_experience(self, state, action, reward, next_state, done):
        # Append a new experience to the repaly buffer
        self.replay_buffer.append((state, self.actions.index(action), reward, next_state, done))

    def replay(self, batch_size):
        # Learns from memorized experience
        minibatch = random.sample(self.replay_buffer, batch_size)
        states, next_states, targets = [], [], []
        # For faster prediction
        for state, action_index, reward, next_state, done in minibatch:
            states.append(state[0])
            next_states.append(next_state[0])

        targets = self.model.predict(np.array(states), verbose=0)
        t = self.target_model.predict(np.array(next_states), verbose=0)

        i = 0
        for state, action_index, reward, next_state, done in minibatch:
            if done:
                targets[i][action_index] = reward
            else:
                targets[i][action_index] = reward + self.gamma * np.amax(t[i])
            i += 1

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        self.update_epsilon()
    
    def load(self, name):
        # Load the neural network`s weights
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        # Saves the neural network`s weights
        self.target_model.save_weights(name)

    def update_epsilon(self):
        # Updates the epsilon used for epsilon-greedy action selection
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay