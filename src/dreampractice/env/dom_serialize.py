"""Serialize MiniWoB++ DOM observations into compact text for an LLM policy.

Design decisions:
- Every interactive element gets a stable integer `ref` (MiniWoB++ provides these).
  The policy addresses elements by ref, never by pixel coordinates. This keeps the
  action space identical between the real env and any structured simulator (S-LLM /
  S-Code), which is what makes sim-trained policies transferable.
- We drop styling and geometry except coarse position, and cap text length.
  Phase 2 "perceptual noise" ablations perturb exactly this serialization.
"""

from __future__ import annotations

from typing import Any

MAX_TEXT_LEN = 60
INTERACTIVE_TAGS = {"button", "input_text", "input_checkbox", "input_radio",
                    "input_password", "input_date", "select", "textarea",
                    "a", "option", "label", "span", "div", "t"}


def _clip(s: str, n: int = MAX_TEXT_LEN) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def element_line(el: dict[str, Any]) -> str | None:
    """One line per element: [ref] tag "text" (attrs)."""
    tag = el.get("tag", "")
    text = _clip(el.get("text", ""))
    ref = el.get("ref")
    if ref is None:
        return None
    attrs = []
    if el.get("value"):
        attrs.append(f'value="{_clip(str(el["value"]))}"')
    if el.get("classes"):
        attrs.append(f'class="{_clip(str(el["classes"]), 30)}"')
    for flag in ("focused", "tampered"):
        if el.get(flag):
            attrs.append(flag)
    attr_s = f" ({', '.join(attrs)})" if attrs else ""
    text_s = f' "{text}"' if text else ""
    return f"[{ref}] {tag}{text_s}{attr_s}"


def serialize_observation(obs: dict[str, Any]) -> str:
    """Full observation -> prompt block. Includes the task utterance."""
    lines = [f"TASK: {obs.get('utterance', '')}", "ELEMENTS:"]
    for el in obs.get("dom_elements", []):
        line = element_line(el)
        if line:
            lines.append(line)
    return "\n".join(lines)


def transition_record(obs: dict, action_str: str, next_obs: dict,
                      reward: float, done: bool) -> dict:
    """Compact (s, a, s', r) record for the trace buffer / simulator training."""
    return {
        "state": serialize_observation(obs),
        "action": action_str,
        "next_state": serialize_observation(next_obs),
        "reward": reward,
        "done": done,
    }
