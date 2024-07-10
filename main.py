import retro
import numpy as np
from utils import SMB

env = retro.make('SuperMarioBros-Nes', state='Level1-1.state', render_mode='human')

env.reset()

ram = env.get_ram()

done = False

while not done:
    env.render()
    action = env.action_space.sample()
    ob, reward, done, fa, info = env.step(action)
    print(SMB.get_tiles_array(SMB.get_tiles(ram)))
    wait = input()
    print("Reward", reward)


