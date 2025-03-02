from pathlib import Path
from pokemon_env import pokemon_env
import matplotlib.pyplot as plt

ep_length = 2048 * 80
sess_id = "runs"
sess_path = Path(sess_id)

env_config = {
  'headless': False, 
  'save_final_state': False, 
  'early_stop': False,
  'action_freq': 5000,
  'init_state': None,
  'max_steps': ep_length, 
  'print_rewards': True,
  'save_video': False,
  'fast_video': True,
  'session_path': sess_path,
  'gb_path': 'pokemon_red.gb',
  'debug': False,
  'reward_scale': 0.5,
  'explore_weight': 0.25,
  'frame_stacks': 1,
}

env = pokemon_env.RedGymEnv(env_config)
observation, info = env.reset()

for i in range(2):
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    env.render()
    print(f"Step: {i}, Reward: {reward}, Done: {done}")
env.close()


print(observation['screens'].shape)
plt.imshow(observation['screens'][...,0], cmap='gray')
plt.show()