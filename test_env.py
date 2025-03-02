from pathlib import Path
from pokemon_env import pokemon_env
from gemini_agent import gemini
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

ep_length = 2048 * 80
sess_id = "runs"
sess_path = Path(sess_id)

env_config = {
  'headless': False, 
  'save_final_state': False, 
  'early_stop': False,
  'action_duration': 1000,
  'init_state': None,
  'max_steps': ep_length, 
  'print_rewards': True,
  'save_video': False,
  'fast_video': False,
  'session_path': sess_path,
  'gb_path': 'pokemon_red.gb',
  'debug': False,
  'reward_scale': 0.5,
  'explore_weight': 0.25,
  'frame_stacks': 1,
}

env = pokemon_env.RedGymEnv(env_config)
agent = gemini.GeminiAgent()
observation, info = env.reset()

for i in range(100):
    action, response = agent.act(observation['screens'][..., 0])
    print(response)
    observation, reward, done, truncated, info = env.step(action)
    env.render()
    print(f"Step: {i}, Reward: {reward}, Done: {done}")
env.close()
