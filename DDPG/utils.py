import cv2
import numpy as np
import random


def noise(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)


def green_mask(observation):
    # convert to hsv
    hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)

    mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))

    # slice the green
    imask_green = mask_green > 0
    green = np.zeros_like(observation, np.uint8)
    green[imask_green] = observation[imask_green]

    return green


def gray_scale(observation):
    gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    return gray


def preprocess_state(observation):
    green = green_mask(observation)
    grey = gray_scale(green)
    state = np.reshape(grey, (grey.shape[0], grey.shape[1], 1))
    return state


def reward_engineering(state, action, reward, next_state, done):
    """
    Makes reward engineering to allow faster training in the Mountain Car environment.

    :param state: state.
    :type state: NumPy array with dimension (1, 2).
    :param action: action.
    :type action: int.
    :param reward: original reward.
    :type reward: float.
    :param next_state: next state.
    :type next_state: NumPy array with dimension (1, 2).
    :param done: if the simulation is over after this experience.
    :type done: bool.
    :return: modified reward for faster training.
    :rtype: float.
    """

    return reward
