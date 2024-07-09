import retro

env = retro.make('SuperMarioBros-Nes', state='Level1-1.state', render_mode='human')

env.reset()

done = False

while not done:
    env.render()
    action = env.action_space.sample()
    ob, reward, done, fa, info = env.step(action)
    print("Reward", reward)


