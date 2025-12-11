from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
import torch
import numpy as np

class MaskInspectionCallback(BaseCallback):
	def __init__(self):
		super().__init__()
		self.original_method = None
	def _on_step(self):
		return True

	def _on_training_start(self):
	# Monkey patch policy forward
		policy = self.model.policy
		self.original_method = policy._get_action_dist_from_latent
	def patched_action_dist(latent_pi, latent_vf, action_masks=None):
		# Inspect masks HERE before SB3 applies them
		if action_masks is not None:
			mask = action_masks.detach().cpu().numpy()
			batch, action_dim = mask.shape
			# === Check A: shape mismatch ===
			expected = latent_pi.shape[-1]
			if action_dim != expected:
				print("\n[ERROR] MASK SHAPE MISMATCH")
				print("mask:", mask.shape)
				print("logits:", latent_pi.shape)
				raise ValueError("Mask size mismatch")
			# === Check B: mask all false ===
			if np.any(mask.sum(axis=1) == 0):
				print("\n[ERROR] MASK ALL FALSE DETECTED")
				print(mask)
				env = self.training_env.envs[0].unwrapped
				print("FEN:", env.board.fen())
				raise ValueError("Invalid mask: all false")
			# === Check C: NaNs ===
			if np.isnan(mask).any():
				print("\n[ERROR] MASK CONTAINS NaN")
				print(mask)
				raise ValueError("Mask NaN!")
