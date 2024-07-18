from utils import test_NN, preprocess_state, select_action, append_result
from collections import deque


def train_ppo(agent, episodes=1000, max_steps=1000):
    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_old_action_means = []
    for episode in range(episodes):
        state, _ = agent.env.reset()
        agent.env.render()
        lastNrewards = deque(maxlen=100)
        result = 0
        for t in range(max_steps):
            state_processed = preprocess_state(state)
            action_mean, old_action_mean = select_action(agent.policy, state_processed)
            temp_action = action_mean.ravel()
            temp_action[-1] = min(temp_action[-1], 0.1)
            temp_action[-2] = max(temp_action[-2], 0.2)

            action = [0, 0, 0]
            action[0] = temp_action[1] - temp_action[0]
            action[1] = temp_action[2]
            action[2] = temp_action[3]

            reward = 0
            for _ in range(3):
                next_state, r, terminated, truncated, _ = env.step(action)
                lastNrewards.append(r)
                reward += r
                done = terminated or truncated
                if done:
                    break
            result += reward

            episode_states.append(state_processed[0])
            episode_actions.append(temp_action)
            episode_rewards.append(reward)
            episode_old_action_means.append(old_action_mean[0])

            state = next_state
            agent.env.render()

            if all(a < 0 for a in lastNrewards):
                break

        print(f"Iteração {episode}\n")

        agent.optimize_policy(episode_states, episode_actions, episode_rewards, episode_old_action_means)
        agent.save_model("temp")
        append_result(result)
        #test_NN(agent)

if __name__ == "__main__":
    import gymnasium as gym
    from ppo_agent import PPOAgent

    env = gym.make("CarRacing-v2", render_mode="human")
    agent = PPOAgent(env)
    agent.load_model("temp")
    #test_NN(agent, 1)
    train_ppo(agent)
