"""Rollout collection against anything that quacks like an environment.

The whole architecture hinges on one Protocol: `TextEnv`. The real MiniWoB++
wrapper, the S-LLM simulator, and the S-Code simulator all implement it, so
`collect_episode` is the single code path for real exploration (step 1 of the
loop), dreamed practice (step 3), and evaluation (step 5). If you find
yourself writing a second rollout function, something has gone wrong.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from ..agent.policy import Policy


class TextEnv(Protocol):
    def reset(self, seed: int | None = None) -> str: ...
    def step(self, action_str: str) -> tuple[str, float, bool, dict]: ...


def collect_episode(env: TextEnv, policy: Policy, seed: int | None = None,
                    max_steps: int = 15, greedy: bool = False) -> dict:
    """Run one episode; return a serializable trajectory dict."""
    obs = env.reset(seed=seed)
    history: list[tuple[str, str]] = []
    traj = {"seed": seed, "turns": [], "total_reward": 0.0, "success": False}

    for _ in range(max_steps):
        action = policy.act(history, obs, greedy=greedy)
        next_obs, reward, done, info = env.step(action)
        traj["turns"].append({
            "obs": obs, "action": action, "reward": reward,
            "parsed": info.get("parsed", True),
            "prompt": policy.build_prompt(history, obs),  # for exact logprob replay
        })
        traj["total_reward"] += reward
        history.append((obs, action))
        obs = next_obs
        if done:
            break

    traj["success"] = traj["total_reward"] > 0.5  # MiniWoB success reward ≈ 1.0
    return traj


def collect_group(env_factory, policy: Policy, task_seed: int,
                  group_size: int = 8, max_steps: int = 15) -> list[dict]:
    """GRPO group: G rollouts of the SAME task instance (same seed => same
    utterance/layout), different sampling. Group-relative advantages only make
    sense if the group shares the task instance."""
    out = []
    for _ in range(group_size):
        env = env_factory()
        try:
            out.append(collect_episode(env, policy, seed=task_seed,
                                       max_steps=max_steps, greedy=False))
        finally:
            close = getattr(env, "close", None)
            if close:
                close()
    return out


def save_trajectories(trajs: list[dict], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for t in trajs:
            f.write(json.dumps(t) + "\n")
