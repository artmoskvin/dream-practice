"""Evaluation: success rate on held-out task instances, greedy decoding,
fixed seed sets. This is the ONLY number that counts — everything upstream
(sim rewards, dreamed success rates) is diagnostics.

Seed discipline: eval seeds are disjoint from all training/exploration seeds
and frozen in configs/base.yaml. Never let a training seed leak into eval —
MiniWoB++ seeds determine the task instance, so leakage = train/test overlap.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..agent.policy import Policy
from ..env.miniwob_env import MiniWobTextEnv
from ..train.rollout import collect_episode


def evaluate(policy: Policy, task_names: list[str], eval_seeds: list[int],
             max_steps: int = 15, out_path: str | Path | None = None) -> dict:
    results = {"per_task": {}, "overall": None, "timestamp": time.time()}
    all_success = []

    for task in task_names:
        env = MiniWobTextEnv(task, max_steps=max_steps)
        succ, parse_fail, steps = [], 0, []
        try:
            for seed in eval_seeds:
                traj = collect_episode(env, policy, seed=seed,
                                       max_steps=max_steps, greedy=True)
                succ.append(traj["success"])
                steps.append(len(traj["turns"]))
                parse_fail += sum(1 for t in traj["turns"] if not t["parsed"])
        finally:
            env.close()
        rate = sum(succ) / len(succ)
        results["per_task"][task] = {
            "success_rate": rate,
            "n": len(succ),
            "mean_steps": sum(steps) / len(steps),
            "parse_failures": parse_fail,
        }
        all_success.extend(succ)
        print(f"[eval] {task}: {rate:.2%} ({sum(succ)}/{len(succ)})")

    results["overall"] = sum(all_success) / len(all_success)
    print(f"[eval] OVERALL: {results['overall']:.2%}")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(results, indent=2))
    return results
