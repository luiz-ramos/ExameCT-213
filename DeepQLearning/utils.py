import cv2
import numpy as np

def process_deque(state_stack):
    # Process the given deque into an array of frames compatible with Keras
    stack_frame = np.array(state_stack)
    stack_frame = np.transpose(stack_frame, (1, 2, 0))
    stack_frame = np.expand_dims(stack_frame, axis=0)

    return stack_frame

def preprocess_state(state):
    # Process the given image simplifying it
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.GaussianBlur(state, (5, 5), 0)
    state = state.astype(float)
    state /= 255.0

    return state

def reward_engineering_car_racing(action, reward):
    # Given extra reward to faster learning
    if action[1] > 0: # action to acelerate
        reward += 1.5

    return reward
