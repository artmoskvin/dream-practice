"""S-LLM: an API LLM as the transition function, grounded in observed traces.

Implements the same TextEnv protocol as MiniWobTextEnv, so the GRPO trainer
and rollout collector are reused unchanged for dreaming. This file is the
Phase-1 workhorse and the Phase-2 corruption target.

The two-hour smoke test lives here: before any training, call
`SLLMSimulator.step()` by hand on a few tasks and read the predicted
transitions. If they're incoherent with good prompting + retrieved traces,
the premise needs rethinking before you spend GPU-weeks on it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

SIM_SYSTEM = """You simulate a web page's response to user actions.
Given the current page state (TASK + ELEMENTS with [ref] ids), an action, and
example transitions from the real page, output the NEXT page state in the
identical ELEMENTS format, then a reward line.

Rules:
- Preserve ref numbering for unchanged elements; new elements get new refs.
- Output format, nothing else:
NEXT_STATE:
<elements block>
REWARD: <float>   # 1.0 only if the TASK is now completed, else 0.0
DONE: <true|false>"""


@dataclass
class TraceBuffer:
    """(s, a, s') records from real exploration; retrieval grounds the sim."""
    records: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "TraceBuffer":
        recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        return cls(records=recs)

    def retrieve(self, state: str, action: str, k: int = 4) -> list[dict]:
        """Toy lexical retrieval — replace with embeddings if it underperforms.
        Score = token overlap on (state + action)."""
        q = set((state + " " + action).split())
        scored = sorted(
            self.records,
            key=lambda r: -len(q & set((r["state"] + " " + r["action"]).split())),
        )
        return scored[:k]


class SLLMSimulator:
    """TextEnv-compatible dreamed environment for ONE task family."""

    def __init__(self, llm_client, buffer: TraceBuffer,
                 initial_states: list[str], max_steps: int = 15,
                 transition_noise_p: float = 0.0):
        """llm_client: callable(messages: list[dict]) -> str  (your API of choice)
        initial_states: serialized first observations from real episodes;
            reset() samples one — the simulator never invents the *start* state.
        transition_noise_p: Phase-2 knob — probability of corrupting a
            transition with a plausible-but-wrong next state."""
        self.llm = llm_client
        self.buffer = buffer
        self.initial_states = initial_states
        self.max_steps = max_steps
        self.noise_p = transition_noise_p
        self._state: str = ""
        self._steps = 0

    def reset(self, seed: int | None = None) -> str:
        rng = random.Random(seed)
        self._state = rng.choice(self.initial_states)
        self._steps = 0
        return self._state

    def step(self, action_str: str) -> tuple[str, float, bool, dict]:
        self._steps += 1
        examples = self.buffer.retrieve(self._state, action_str)
        ex_block = "\n\n".join(
            f"EXAMPLE:\nSTATE:\n{e['state']}\nACTION: {e['action']}\n"
            f"NEXT:\n{e['next_state']}" for e in examples)
        messages = [
            {"role": "system", "content": SIM_SYSTEM},
            {"role": "user", "content":
                f"{ex_block}\n\nCURRENT STATE:\n{self._state}\n"
                f"ACTION: {action_str}\n\nSimulate."},
        ]
        raw = self.llm(messages)
        next_state, reward, done = self._parse(raw)

        if self.noise_p and random.random() < self.noise_p:
            # Phase-2 transition-logic corruption: swap in a retrieved but
            # WRONG next state (from a different transition).
            wrong = random.choice(self.buffer.records)
            next_state = wrong["next_state"]

        self._state = next_state
        done = done or self._steps >= self.max_steps
        return next_state, reward, done, {"sim": True}

    @staticmethod
    def _parse(raw: str) -> tuple[str, float, bool]:
        state, reward, done = raw, 0.0, False
        if "NEXT_STATE:" in raw:
            body = raw.split("NEXT_STATE:", 1)[1]
            if "REWARD:" in body:
                state, rest = body.split("REWARD:", 1)
                try:
                    reward = float(rest.strip().split()[0])
                except (ValueError, IndexError):
                    reward = 0.0
                done = "DONE: true" in rest.lower().replace("done:", "DONE:")
            else:
                state = body
        return state.strip(), reward, done
