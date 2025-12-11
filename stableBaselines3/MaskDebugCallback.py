from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch


class MaskDebugCallback(BaseCallback):
    """
    Debug callback for MaskablePPO.
    Stops training immediately when a mask / prob simplex error is detected.
    """

    def __init__(self, verbose=1):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Get the rollout buffer last obs & mask (SB3 stores mask in infos["action_mask"])
        try:
            last_obs = self.locals["rollout_buffer"].observations[-1]
            last_info = self.locals["infos"][-1]
        except Exception:
            return True  # skip very early steps

        if "action_mask" not in last_info:
            return True  # mask not used yet

        mask = last_info["action_mask"]
        mask = np.asarray(mask, dtype=bool)

        # === 1. Check mask correctness ===
        if mask.ndim != 1:
            print("\n[MaskDebug] ERROR: Mask is not 1-dimensional:", mask.shape)
            return False

        if mask.sum() == 0:
            print("\n[MaskDebug] ERROR: Mask has NO legal actions!")
            print("Mask:", mask)
            print("Env state FEN:", self.training_env.envs[0].unwrapped.board.fen())
            return False

        # === 2. Check model raw logits ===
        policy: MaskableActorCriticPolicy = self.model.policy

        obs_tensor = torch.as_tensor(last_obs, device=policy.device)
        dist = policy.get_distribution(obs_tensor)

        logits = dist.distribution.logits.detach().cpu().numpy()

        if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
            print("\n[MaskDebug] ERROR: NaN or Inf in logits!")
            print("Logits:", logits)
            print("Mask:", mask)
            print("Env FEN:", self.training_env.envs[0].unwrapped.board.fen())
            return False

        # === 3. Check masked probs ===
        probs = dist.distribution.probs.detach().cpu().numpy()

        if probs.shape[-1] != mask.shape[-1]:
            print("\n[MaskDebug] ERROR: Mask length does not match probs length!")
            print("Probs shape:", probs.shape)
            print("Mask shape:", mask.shape)
            return False

        # values outside simplex means probs < 0, > 1 or sum != 1
        if np.any(probs < -1e-7) or np.any(probs > 1 + 1e-7):
            print("\n[MaskDebug] ERROR: Probabilities out of simplex bounds.")
            print("Min prob:", probs.min(), "Max prob:", probs.max())
            print("Logits:", logits)
            print("Mask:", mask)
            return False

        prob_sum = probs.sum()
        if not np.isclose(prob_sum, 1, atol=1e-4):
            print("\n[MaskDebug] ERROR: Probability sum ≠ 1:", prob_sum)
            print("Probs:", probs)
            print("Mask:", mask)
            return False

        return True

