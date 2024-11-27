import gymnasium
import numpy as np

# Configuration of the enviroment
env = gymnasium.make('Ant-v5', xml_file='./mujoco_menagerie/unitree_go1/scene.xml',render_mode='human')

#Info about the Env
observation, info = env.reset()
print("Espacio de observaciones:", env.observation_space)
print("Espacio de acciones:", env.action_space)

## TRAINING ###

env.close