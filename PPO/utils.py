import numpy as np
from collections import deque
import csv
import os
import matplotlib.pyplot as plt
def preprocess_state(state):
    observation = np.reshape(state, (1, 96, 96, 3))
    observation = observation / 255.0

    return observation


def select_action(policy, state):
    old_action_mean = policy.predict(state, verbose=0)
    noise = np.random.normal(loc=0, scale=0.1, size=old_action_mean.shape)
    noise[0][-2] /= 4
    noise[0][-2] += 0.2
    noise[0][-1] /= 4
    action_mean = old_action_mean + noise
    action_mean = np.clip(action_mean, 0, 1)

    return action_mean, old_action_mean


def test_NN(agent, n):
    result = []
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
            temp_action[-2] = max(temp_action[-2], 0.1)

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
        result.append(reward)
    return result


def append_result(total_reward):
    file_path = 'results.csv'
    # Check if file exists to determine if header is needed
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write the header if the file doesn't exist
            writer.writerow(['iteration', 'total_reward'])

        # Get the current iteration (number of lines in the file)
        iteration = sum(1 for row in open(file_path)) - 1
        # Append the result
        writer.writerow([iteration, total_reward])


def plot_result(file_path='results.csv'):

    if not os.path.isfile(file_path):
        print("Results file not found.")
        return

    iterations = []
    total_rewards = []

    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            iterations.append(int(row['iteration']))
            total_rewards.append(float(row['total_reward']))

    plt.figure(figsize=(10, 5))
    plt.plot(iterations, total_rewards, marker='o')
    plt.title('Total Reward vs. Episode')
    plt.xlabel('Episode')
    plt.ylabel('Train Total Reward x Episode')
    plt.grid(True)
    plt.show()
