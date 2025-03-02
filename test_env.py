from pathlib import Path
from pokemon_env import pokemon_env
from gemini_agent import gemini
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import time

load_dotenv()

ep_length = 1000
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
  'save_video': True,
  'fast_video': True,
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

start_time = time.time()
truncated, step = False, 0
while not truncated:
    step += 1
    action, response = agent.act(observation['screens'][..., 0])
    print(response)
    observation, reward, done, truncated, info = env.step(action)
    env.render()
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Step: {step}, Reward: {reward}, Done: {truncated}, Time Elapsed: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
env.close()
