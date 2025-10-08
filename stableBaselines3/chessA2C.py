import gymnasium as gym
from stable_baselines3 import A2C
import chess_gym


#create the enviroment
env = gym.make("Chess-v0", render_mode="human", observation_mode="piece_map")

#Try to get a list of legal actions for masking (i think not possible on A2C) | MaskedPPO seems to be a thing on AB3 but cant import it
#Cant access the function
env._get_legal_moves_index()

#create and train the model
#Currently unable to trains as illegal moves are chosen as actions
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

#reset the enviroment state
vec_env = model.get_env()
obs = vec_env.reset()
