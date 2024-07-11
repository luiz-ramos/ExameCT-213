import gymnasium as gym

env = gym.make("CarRacing-v2", render_mode='human')
state_size = env.observation_space.shape
print(state_size)

env.reset()

done = False

while not done:
    env.render()
    action = env.action_space.sample()
    ob, reward, done, fa, info = env.step(action)
    print("Observation", env.observation_space)


