import gymnasium as gym

env = gym.make("CarRacing-v2", render_mode='human')

env.reset()

done = False

while not done:
    env.render()
    action = env.action_space.sample()
    ob, reward, done, fa, info = env.step(action)
    print("Reward", reward)


