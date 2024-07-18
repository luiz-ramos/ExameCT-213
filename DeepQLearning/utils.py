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

def test_nn(agent, env, n, stack_frames = 3, skip_frames = 3):
    done = False
    total_return_history, cumulative_return_history = [], []

    for episodes in range(1, n+1):
        # Reset the environment
        state, _ = env.reset()
        state = preprocess_state(state)
        state_stack = deque([state]*stack_frames, maxlen=stack_frames)
        
        total_reward = 0.0
        cumulative_reward = 0.0
        done = False
        
        for time in range(1, 180):
            curr_state_stack = process_deque(state_stack)
            action = agent.act(curr_state_stack)

            reward = 0
            for _ in range(skip_frames+1):
                next_state, r, terminated, truncated, _ = env.step(action)
                reward += r
                done = terminated or truncated
                if done:
                    break

            total_reward += reward
            cumulative_reward = agent.gamma * cumulative_reward + reward

            next_state = preprocess_state(next_state)
            state_stack.append(next_state)

            if done or time == 179:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards(adjusted): {:.2}, Cumulative Rewards: {:.2}'.format(episodes, n, time, float(total_reward), float(cumulative_reward)))
                total_return_history.append(total_reward)
                cumulative_return_history.append(cumulative_reward)
                break
    return total_return_history, cumulative_return_history
