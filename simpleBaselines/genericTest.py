import gymnasium as gym
import matplotlib.pyplot as plt
import chess_gym

# Select rendering mode
rendermode = input("select rendering mode: human | training: \n")
#rendermode = "human"

# Select RL algorithm
algorithm = input("Select RL algorithm: \nQlearning\n")

match algorithm:
	case "Qlearning":
		#Call the qlearning training script



