"""Minimal GRPO for multi-turn episodes on a single GPU.

Why hand-rolled instead of TRL's GRPOTrainer: TRL's implementation assumes
single-turn prompt->completion. Our unit of reward is the episode (multi-turn,
env in the loop), so we collect full trajectories ourselves and apply the
group-relative advantage to every action in the episode. This is the simplest
credit assignment that can work (episode-level, no per-step shaping) — same
spirit as DynaWeb's sequence-level optimization.

Recipe per update:
  1. Pick a task instance (task_name, seed).
  2. collect_group(): G rollouts, same instance, temperature sampling.
  3. reward r_i = success (1/0) + small step/parse penalties already in-env.
  4. advantage A_i = (r_i - mean(r)) / (std(r) + eps). If std == 0 the group
     is uninformative (all succeed / all fail) -> skip. Track the skip rate:
     if >80% of groups are skipped, your task selection has no learning signal
     at this policy strength (curriculum problem, not an RL bug).
  5. loss = - (1/G) Σ_i A_i · Σ_turns logprob(action | prompt)
  6. Optional KL penalty to the frozen base (adapters disabled) — start
     without it; add if the policy degenerates into parse failures.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..agent.policy import Policy
from .rollout import collect_group


@dataclass
class GRPOConfig:
    group_size: int = 8
    lr: float = 1e-5
    grad_accum_groups: int = 2      # groups per optimizer step
    max_grad_norm: float = 1.0
    adv_eps: float = 1e-6
    updates: int = 200


class GRPOTrainer:
    def __init__(self, policy: Policy, cfg: GRPOConfig):
        self.policy = policy
        self.cfg = cfg
        trainable = [p for p in policy.model.parameters() if p.requires_grad]
        try:
            import bitsandbytes as bnb
            self.opt = bnb.optim.PagedAdamW8bit(trainable, lr=cfg.lr)
        except ImportError:
            self.opt = torch.optim.AdamW(trainable, lr=cfg.lr)
        self.stats = {"groups": 0, "skipped": 0, "mean_reward": []}

    def group_loss(self, trajs: list[dict]) -> torch.Tensor | None:
        rewards = torch.tensor([t["total_reward"] for t in trajs])
        self.stats["mean_reward"].append(rewards.mean().item())
        if rewards.std() < self.cfg.adv_eps:
            return None  # uninformative group
        adv = (rewards - rewards.mean()) / (rewards.std() + self.cfg.adv_eps)

        loss = torch.zeros((), device=self.policy.model.device)
        for traj, a in zip(trajs, adv):
            ep_lp = torch.zeros((), device=self.policy.model.device)
            for turn in traj["turns"]:
                ep_lp = ep_lp + self.policy.action_logprob(
                    turn["prompt"], turn["action"])
            loss = loss - a.to(loss.device) * ep_lp
        return loss / len(trajs)

    def train(self, env_factory, task_seeds: list[int], max_steps: int = 15):
        self.policy.model.train()
        accum = 0
        for step in range(self.cfg.updates):
            seed = task_seeds[step % len(task_seeds)]
            trajs = collect_group(env_factory, self.policy, task_seed=seed,
                                  group_size=self.cfg.group_size,
                                  max_steps=max_steps)
            self.stats["groups"] += 1
            loss = self.group_loss(trajs)
            if loss is None:
                self.stats["skipped"] += 1
                continue
            (loss / self.cfg.grad_accum_groups).backward()
            accum += 1
            if accum >= self.cfg.grad_accum_groups:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy.model.parameters()
                     if p.requires_grad],
                    self.cfg.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                accum = 0
            if step % 10 == 0:
                mr = sum(self.stats["mean_reward"][-10:]) / max(
                    1, len(self.stats["mean_reward"][-10:]))
                skip = self.stats["skipped"] / max(1, self.stats["groups"])
                print(f"[grpo] step={step} mean_r(10)={mr:.3f} "
                      f"skip_rate={skip:.2f}")
