# dream-practice — Phase 0 harness

Scaffold for the *Dream-and-Practice* experiments (see the research proposal):
can a GUI agent improve on a specific environment by building a simulator from
its own traces and practicing in it?

## Architecture in one sentence

Everything — the real MiniWoB++ env, the S-LLM dreamed env, and (later)
S-Code clones — implements one `TextEnv` protocol (`reset/step` over
serialized DOM text), so a single rollout collector, a single GRPO trainer,
and a single eval runner cover all six experimental conditions.

```
src/dreampractice/
├── env/
│   ├── dom_serialize.py   # obs -> compact text; Phase-2 perceptual-noise target
│   └── miniwob_env.py     # real env; CLICK/TYPE/STOP grammar; episode logging
├── agent/
│   ├── prompts.py         # frozen across conditions — never tune per-condition
│   └── policy.py          # Qwen2.5-7B NF4 + LoRA r16; act() + action_logprob()
├── sim/
│   └── s_llm.py           # TraceBuffer + API-LLM transition model (TextEnv)
├── train/
│   ├── rollout.py         # THE shared code path: real, dreamed, and eval rollouts
│   └── grpo.py            # episode-level group-relative PG, hand-rolled
└── eval/
    └── runner.py          # greedy, frozen disjoint seeds, JSON output
```

## Setup (Linux, single RTX 3090)

```bash
python -m venv .venv && source .venv/bin/activate
pip install "gymnasium>=0.29" miniwob selenium torch transformers peft \
            bitsandbytes accelerate pyyaml
# MiniWoB++ needs Chrome/Chromium + chromedriver on PATH (headless is fine).
# On Arch: pacman -S chromium  (chromedriver ships with it)
```

## Week-one checklist (in order — each step de-risks the next)

1. **Env smoke test.** Random-action rollouts on `click-button`. Confirms
   Selenium/Chrome plumbing, the action grammar mapping, and reward wiring.
   Expected pain point: `create_action` signatures in `miniwob_env.py` vs.
   your installed miniwob version. Budget 20 minutes for the fix.
2. **C0 baseline.** Frozen Qwen2.5-7B, greedy, 40 eval seeds, all 12 task
   families. You need base success in the 10–70% band on train families —
   prune tasks outside it (no headroom = no signal). Log parse-failure rate;
   if >10%, tighten the grammar prompt before anything else.
3. **The two-hour premise test.** Collect 10 real episodes on `login-user`,
   load them into `TraceBuffer`, and manually drive `SLLMSimulator.step()`
   with a strong API model. Read the predicted transitions. Coherent →
   proceed. Garbage → the S-LLM branch is in trouble; move S-Code up the
   priority list before burning GPU-weeks.
4. **GRPO sanity run.** Direct RL (condition C3) on ONE easy family
   (`click-button`), 50 updates. You're looking for: mean reward trending up,
   skip-rate < 0.8, parse failures not exploding. This validates the trainer
   with zero simulator complexity in the loop.
5. Only then: wire the full C4 pipeline (explore → build → dream → distill →
   eval) end-to-end on one family.

## VRAM expectations (verify empirically at step 2)

NF4 7B weights ~4.7 GB · LoRA r16 trainable ~40 MB · KV @4k bf16 ~1 GB ·
paged 8-bit AdamW on LoRA only · gradient checkpointing on. Training and
generation coexist on 24 GB with room to spare at bs=1. If `action_logprob`
replay OOMs on long episodes, process turns with per-turn backward
(accumulate grads turn-by-turn) instead of summing the episode first.

## Invariants — do not violate

- Eval seeds (9000+) never appear in exploration or training. MiniWoB++
  seeds define task instances; leakage = train-on-test.
- `prompts.py` is identical for every condition.
- Dreamed sample counts are unlimited; REAL sample counts are the budget B.
  Report both, always.
- The simulator's reward is never trusted as a result — only
  `eval/runner.py` numbers on the real env count.

## What's deliberately missing (later phases)

- `sim/s_code.py` — coding-agent-generated executable clones (Phase 1.5;
  same TextEnv protocol, near-zero cost per rollout).
- `train/sft.py` — rejection-sampling SFT (fallback if GRPO is unstable:
  best-of-n dreamed trajectories, standard SFT on winners).
- OPSD distillation arm; DreamGym-style curriculum generation; Phase-2
  corruption sweep driver (the `transition_noise_p` knob already exists).
