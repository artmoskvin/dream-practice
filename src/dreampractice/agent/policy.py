"""LLM policy: 4-bit base + LoRA adapters, one action per generate() call.

3090 (24 GB) budget notes, measured mentally not empirically — verify on day one:
  - Qwen2.5-7B-Instruct NF4 weights ............ ~4.7 GB
  - LoRA (r=16, qkvo+gate/up/down) ............. ~40 MB trainable
  - KV cache @ 4k ctx, bf16, bs=1 .............. ~1 GB
  - Optimizer (paged_adamw_8bit on LoRA only) .. small
  - Activations w/ gradient checkpointing ...... the remaining headroom
  Training and generation share the GPU; generate with adapters enabled
  (on-policy) and torch.no_grad().
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .prompts import SYSTEM_PROMPT, format_turn


@dataclass
class PolicyConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 48        # actions are short; cap hard
    temperature: float = 0.7        # >0 for GRPO group diversity; 0 for eval
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_context_tokens: int = 4096


class Policy:
    def __init__(self, cfg: PolicyConfig, train_mode: bool = True):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, quantization_config=bnb, device_map="auto",
            attn_implementation="sdpa",
        )
        if train_mode:
            model = prepare_model_for_kbit_training(model)
            lora = LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout, bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
            )
            model = get_peft_model(model, lora)
            model.gradient_checkpointing_enable()
        self.model = model

    # -- inference -----------------------------------------------------------
    def build_prompt(self, history: list[tuple[str, str]], obs_text: str) -> str:
        """history: [(obs_text, action_str), ...] earlier turns."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for past_obs, past_act in history:
            messages.append({"role": "user", "content": format_turn(past_obs)})
            messages.append({"role": "assistant", "content": past_act})
        messages.append({"role": "user", "content": format_turn(obs_text)})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def act(self, history: list[tuple[str, str]], obs_text: str,
            greedy: bool = False) -> str:
        prompt = self.build_prompt(history, obs_text)
        ids = self.tokenizer(prompt, return_tensors="pt",
                             truncation=True,
                             max_length=self.cfg.max_context_tokens
                             ).to(self.model.device)
        out = self.model.generate(
            **ids,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=not greedy,
            temperature=self.cfg.temperature if not greedy else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        return text.strip().splitlines()[0] if text.strip() else "STOP()"

    # -- training-side helper -------------------------------------------------
    def action_logprob(self, prompt: str, action: str) -> torch.Tensor:
        """Sum of token logprobs of `action` given `prompt` (grad-enabled).

        Used by GRPO: loss = -advantage * action_logprob, summed over the
        episode's actions. Prompt tokens are masked out of the loss.
        """
        full = prompt + action
        enc = self.tokenizer(full, return_tensors="pt",
                             truncation=True,
                             max_length=self.cfg.max_context_tokens
                             ).to(self.model.device)
        prompt_len = self.tokenizer(prompt, return_tensors="pt",
                                    truncation=True,
                                    max_length=self.cfg.max_context_tokens
                                    )["input_ids"].shape[1]
        out = self.model(**enc)
        logits = out.logits[:, :-1, :]
        targets = enc["input_ids"][:, 1:]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        action_lp = tok_lp[:, prompt_len - 1:]  # only action tokens
        return action_lp.sum()
