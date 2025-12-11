from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch

class RolloutMaskInspector(BaseCallback):
	def _on_step(self):
		return True
	"""
	Inspects masks *right before PPO training*, so it catches issues that
	appear during the update (not during rollouts).
	"""
	def _on_rollout_end(self) -> None:
		rb = self.model.rollout_buffer
		# Extract the most recent batch of masks from infos
		try:
			infos = rb.infos # list of list-of-dicts [n_steps][n_envs]
		except Exception as e:
			print("[Inspector] Could not access rollout buffer infos:", e)
			return
		num_steps = len(infos)
		num_envs = len(infos[0])
		print(f"[Inspector] Checking rollout buffer: {num_steps} steps × {num_envs} envs")
		for step in range(num_steps):
			for env_i in range(num_envs):
				info = infos[step][env_i]
				if "action_mask" not in info:
					continue
				mask = np.array(info["action_mask"], dtype=bool)
				# === Check 1: Mask shape ===
				if mask.size != self.model.policy.action_dist.distribution.logits.shape[-1]:
					print("\n[FATAL] MASK SIZE MISMATCH BEFORE TRAINING!")
					print("mask size:", mask.size)
					print("logits expected:", self.model.policy.action_dist.distribution.logits.shape[-1])
					print("mask:", mask)
					print("env step:", step)
					self.model.save("broken_model.zip")
					raise ValueError("Mask shape mismatch")
				# === Check 2: Mask all false ===
				if mask.sum() == 0:
					print("\n[FATAL] MASK IS ALL FALSE BEFORE TRAINING!")

					print("mask:", mask)
					fen = self.training_env.envs[env_i].unwrapped.board.fen()
					print("FEN:", fen)
					raise ValueError("Invalid mask: no legal actions")
		print("[Inspector] Rollout masks look OK.")


