"""Gymnasium wrapper around MiniWoB++ exposing a text-in / text-out interface.

The policy emits exactly one action per turn in a constrained grammar:

    CLICK(ref)              e.g.  CLICK(42)
    TYPE(ref, "some text")  e.g.  TYPE(7, "john@example.com")
    STOP()                  give up / declare done

Keeping the grammar tiny matters for two reasons: (1) parse failures become a
clean, loggable event (we count them as no-ops with a small penalty) rather
than silent weirdness, and (2) the same grammar is what the S-LLM/S-Code
simulators consume, so real and dreamed trajectories are format-identical.

NOTE: MiniWoB++ (Farama `miniwob` package) API drifts between versions.
This targets miniwob>=1.0 with gymnasium. Check `env.unwrapped.create_action`
signatures against your installed version on day one — this is the most
likely file to need a 20-minute fix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym

from .dom_serialize import serialize_observation, transition_record

ACTION_RE = re.compile(
    r"^\s*(CLICK|TYPE|STOP)\s*\(\s*(?:(\d+)\s*(?:,\s*\"(.*)\")?)?\s*\)\s*$",
    re.DOTALL,
)

PARSE_FAILURE_PENALTY = -0.05  # small shaping penalty; success reward is ~1.0


@dataclass
class EpisodeLog:
    task: str
    seed: int
    transitions: list[dict] = field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False
    steps: int = 0
    parse_failures: int = 0


class MiniWobTextEnv:
    """Text interface over one MiniWoB++ task."""

    def __init__(self, task_name: str, max_steps: int = 15):
        # e.g. task_name = "click-button" -> env id "miniwob/click-button-v1"
        self.env = gym.make(f"miniwob/{task_name}-v1")
        self.task_name = task_name
        self.max_steps = max_steps
        self._obs: dict | None = None

    # -- lifecycle ---------------------------------------------------------
    def reset(self, seed: int | None = None) -> str:
        self._obs, _info = self.env.reset(seed=seed)
        self._log = EpisodeLog(task=self.task_name, seed=seed if seed is not None else -1)
        return serialize_observation(self._obs)

    def close(self):
        self.env.close()

    # -- action parsing ----------------------------------------------------
    def _to_env_action(self, verb: str, ref: int | None, text: str | None):
        u = self.env.unwrapped
        if verb == "CLICK":
            return u.create_action(u.action_space_config.action_types.index("CLICK_ELEMENT")
                                   if hasattr(u.action_space_config, "action_types") else 0,
                                   ref=ref)
        if verb == "TYPE":
            # Convention: TYPE focuses the element then types the full string.
            return u.create_action(
                u.action_space_config.action_types.index("FOCUS_ELEMENT_AND_TYPE_TEXT")
                if hasattr(u.action_space_config, "action_types") else 1,
                ref=ref, text=text or "")
        raise ValueError(verb)

    # -- step ----------------------------------------------------------------
    def step(self, action_str: str) -> tuple[str, float, bool, dict]:
        """Returns (obs_text, reward, done, info). Parse failures are no-ops."""
        m = ACTION_RE.match(action_str or "")
        self._log.steps += 1
        info: dict[str, Any] = {"parsed": bool(m), "raw_action": action_str}

        if m is None:
            self._log.parse_failures += 1
            done = self._log.steps >= self.max_steps
            return serialize_observation(self._obs), PARSE_FAILURE_PENALTY, done, info

        verb, ref_s, text = m.group(1), m.group(2), m.group(3)
        if verb == "STOP":
            return serialize_observation(self._obs), 0.0, True, info

        env_action = self._to_env_action(verb, int(ref_s) if ref_s else None, text)
        next_obs, reward, terminated, truncated, step_info = self.env.step(env_action)
        done = bool(terminated or truncated or self._log.steps >= self.max_steps)

        self._log.transitions.append(
            transition_record(self._obs, action_str.strip(), next_obs, float(reward), done))
        self._log.total_reward += float(reward)
        if float(reward) > 0 and terminated:
            self._log.success = True
        self._obs = next_obs
        info.update(step_info or {})
        return serialize_observation(next_obs), float(reward), done, info

    @property
    def episode_log(self) -> EpisodeLog:
        return self._log
