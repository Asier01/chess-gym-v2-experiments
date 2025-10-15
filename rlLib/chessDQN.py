from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

import gymnasium as gym
import chess_gym
from pprint import pprint

config = (
    DQNConfig()
    .environment("Chess-v0")
    .training(replay_buffer_config={
        "type": "PrioritizedEpisodeReplayBuffer",
        "capacity": 60000,
        "alpha": 0.5,
        "beta": 0.5,
    })
    .env_runners(num_env_runners=1)
)
algo = config.build_algo()
algo.train()
pprint(algo.evaluate())
algo.stop()
