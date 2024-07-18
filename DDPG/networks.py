import keras
from keras import layers

def get_actor(name, state_shape, action_size):
    # Initialize weights between -3e-3 and 3-e3
    last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

    # State as input
    inputs = layers.Input(shape=state_shape)
    x = inputs
    x = layers.Conv2D(16, kernel_size=(5, 5), strides=(4, 4), padding='valid', use_bias=False, activation="relu")(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(3, 3), padding='valid', use_bias=False, activation="relu")(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(3, 3), padding='valid', use_bias=False, activation="relu")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    y = layers.Dense(action_size, activation='tanh', kernel_initializer=last_init)(x)

    model = keras.Model(inputs=inputs, outputs=y, name=name)
    model.summary()
    return model


def get_critic(name, state_shape, action_size):
    # State as input
    state_inputs = layers.Input(shape=state_shape)
    x = state_inputs
    x = layers.Conv2D(16, kernel_size=(5, 5), strides=(4, 4), padding='valid', use_bias=False, activation="relu")(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(3, 3), padding='valid', use_bias=False, activation="relu")(x)
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(3, 3), padding='valid', use_bias=False, activation="relu")(x)

    x = layers.Flatten()(x)
    action_inputs = layers.Input(shape=(action_size,))
    x = layers.concatenate([x, action_inputs])

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    y = layers.Dense(1)(x)

    model = keras.Model(inputs=[state_inputs, action_inputs], outputs=y, name=name)
    model.summary()
    return model
