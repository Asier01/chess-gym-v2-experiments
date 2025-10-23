import gymnasium as gym
import chess_gym
from stable_baselines3 import DQN

env = gym.make("Chess-v0", render_mode="human", observation_mode="piece_map")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)

#model.save("dqn_chess")

#del model # remove to demonstrate saving and loading

#model = DQN.load("dqn_chess")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
