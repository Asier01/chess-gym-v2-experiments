import gymnasium as gym
import chess_gym
import matplotlib.pyplot as plt

import MaskDebugCallback as mdc
import RolloutMaskInspector as rmi
import MaskInspectionCallback as mic

#from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.results_plotter import plot_results
#from stable_baselines3.common import results_plotter

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

#Define action masking function
def mask_fn(env):
        base_env = env.unwrapped
        return base_env.get_action_mask()


#Initialize enviroment
env = gym.make("Chess-v0", render_mode="training", observation_mode="piece_map", render_steps=True, steps_per_render = 1, reward_type = "dense" , use_eval="stockfish", rival_agent="engine", engine_time_limit = 0.000001, claim_draw=False)
#env = ChessEnv()

#Wrap the enviroment
env = ActionMasker(env, mask_fn)

#log_dir = "."
#env = Monitor(env, log_dir)

model = MaskablePPO("MlpPolicy", env, verbose=2, learning_rate=0.003 , n_steps= 10)

model.learn(total_timesteps=150000, log_interval=10)  #,callback = mic.MaskInspectionCallback())

#plot_results([log_dir], 20_000, results_plotter.X_EPISODES, "MaskedPPO Chess")
#plt.show()

obs, info = env.reset()


'''
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward)
    if terminated or truncated:
        obs, info = env.reset()
'''

