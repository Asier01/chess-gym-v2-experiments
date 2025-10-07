import gymnasium as gym
import matplotlib.pyplot as plt
import chess_gym
import torch
from SimpleBaselines.agent.rl_agents.QLearning_RL_Agent import QLearning_RL_Agent


# Create the environment
env = gym.make("Chess-v0", render_mode="human", observation_mode="piece_map")
#env = gym.make("Chess-v0", render_mode="human", observation_mode="rgb_array")

# Analyze the environment
print('Observation space:', env.observation_space)
print('Observation space sample:', env.observation_space.sample())
print('Action space:', env.action_space)
print('Action space sample:', env.action_space.sample())

# experiment hyperparameters
num_episodes = 1
seed = 126

# experiment results
rewards_total = list()
steps_total = list()

# Create the agent
QLearning_agent = QLearning_RL_Agent(env=env, seed=seed)

# Play the game
for episode in range(num_episodes):

	QLearning_agent.reset_env(seed=seed)

	# Play the game (new environment for each run with continuously learning agent)
	QLearning_agent.play(max_steps=5000, seed=seed)

	rewards_total.append(QLearning_agent.final_state.cumulative_reward)
	steps_total.append(QLearning_agent.final_state.step + 1)
	egreedy_total.append(QLearning_agent.egreedy)

env.close()


print("Percent of episodes finished successfully: {}".format(sum(rewards_total)/num_episodes))
print("Percent of episodes finished successfully (last 100 episodes): {}".format(sum(rewards_total[-100:])/100))

print("Average number of steps: {}".format(sum(steps_total)/num_episodes))
print("Average number of steps (last 100 episodes): {}".format(sum(steps_total[-100:])/100))


