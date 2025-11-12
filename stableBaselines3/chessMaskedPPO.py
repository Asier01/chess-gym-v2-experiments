import gymnasium as gym
import chess_gym

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

#Define action masking function
def mask_fn(env):
        base_env = env.unwrapped
        return base_env.get_action_mask()


#Initialize enviroment
env = gym.make("Chess-v0", render_mode="human", observation_mode="piece_map", render_steps=False, steps_per_render = 10)
#env = ChessEnv()

#Wrap the enviroment
env = ActionMasker(env, mask_fn)


model = MaskablePPO("MlpPolicy", env, verbose=2)

model.learn(total_timesteps=50000, log_interval=10)

obs, info = env.reset()

for i in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()


