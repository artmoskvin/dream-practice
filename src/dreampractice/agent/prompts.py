"""Prompt templates. Deliberately minimal — the policy is supposed to learn,
not be prompt-engineered into competence. Keep this frozen across all
conditions (C0–C5) so prompt quality never confounds the comparison.
"""

SYSTEM_PROMPT = """You are a web automation agent operating a browser UI.
Each turn you see the TASK and the current page ELEMENTS, each with a [ref] number.
Respond with EXACTLY ONE action and nothing else:
  CLICK(ref)
  TYPE(ref, "text")
  STOP()
Use STOP() only when the task is complete or impossible."""


def format_turn(obs_text: str) -> str:
    return obs_text + "\n\nYour action:"
