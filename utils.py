import cv2
import numpy as np
from collections import deque
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


def select_action(policy, state):
    old_action_mean = policy.predict(state, verbose=0)
    noise = np.random.normal(loc=0, scale=0.1, size=old_action_mean.shape)
    noise[0][-2] /= 4
    noise[0][-1] *= 3
    action_mean = old_action_mean + noise
    action_mean = np.clip(action_mean, 0, 1)

    return action_mean, old_action_mean


def test_NN(agent, n):
    result = 0
    for _ in range(n):
        done = False
        state, _ = agent.env.reset()
        reward = 0
        lastNrewards = deque(maxlen=100)
        while not done:
            state_processed = preprocess_state(state)
            action = agent.policy.predict(state_processed, verbose=0)
            temp_action = action.ravel()

            temp_action[-1] = min(temp_action[-1], 0.1)
            action = [0, 0, 0]
            action[0] = temp_action[1] - temp_action[0]
            action[1: 2] = temp_action[2: 3]

            for _ in range(3):
                next_state, r, terminated, truncated, _ = agent.env.step(action)
                lastNrewards.append(r)
                reward += r
                done = terminated or truncated
                if done:
                    break

            agent.env.render()
            state = next_state

            if all(a < 0 for a in lastNrewards):
                break
        result += reward
    return result

