from __future__ import annotations

import torch
from torch import nn
from transformers import OPTConfig, OPTForCausalLM


def build_model(config: dict, seed: int) -> nn.Module:
    torch.manual_seed(int(seed))
    architecture = str(config.get("architecture", "opt"))
    if architecture != "opt":
        raise ValueError(f"unsupported model architecture: {architecture}")

    opt_config = OPTConfig(
        vocab_size=int(config["vocab_size"]),
        hidden_size=int(config["hidden_size"]),
        ffn_dim=int(config["ffn_dim"]),
        num_hidden_layers=int(config["num_hidden_layers"]),
        num_attention_heads=int(config["num_attention_heads"]),
        max_position_embeddings=int(config["max_position_embeddings"]),
        word_embed_proj_dim=int(config["word_embed_proj_dim"]),
        activation_function=str(config["activation_function"]),
        dropout=float(config["dropout"]),
        attention_dropout=float(config["attention_dropout"]),
        activation_dropout=float(config["activation_dropout"]),
        layerdrop=float(config["layerdrop"]),
        do_layer_norm_before=bool(config["do_layer_norm_before"]),
        enable_bias=bool(config["enable_bias"]),
        init_std=float(config["init_std"]),
        bos_token_id=int(config["bos_token_id"]),
        eos_token_id=int(config["eos_token_id"]),
        pad_token_id=int(config["pad_token_id"]),
        use_cache=False,
    )
    model = OPTForCausalLM(opt_config)
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    _verify_parameter_count(model, config)
    return model


def _verify_parameter_count(model: nn.Module, config: dict) -> None:
    expected = config.get("expected_parameter_count")
    if expected is None:
        return
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual != int(expected):
        raise ValueError(
            f"model parameter count mismatch: expected {int(expected)}, got {actual}"
        )


def should_decay(name: str, parameter: nn.Parameter) -> bool:
    lowered = name.lower()
    if parameter.ndim < 2 or name.endswith(".bias"):
        return False
    if "layer_norm" in lowered or "layernorm" in lowered or ".ln_" in lowered:
        return False
    return not ("embed" in lowered or lowered.endswith("lm_head.weight"))


def build_adamw(model: nn.Module, config: dict) -> torch.optim.AdamW:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (decay if should_decay(name, parameter) else no_decay).append(parameter)
    return torch.optim.AdamW(
        (
            {"params": decay, "weight_decay": float(config["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ),
        lr=float(config["learning_rate"]),
        betas=tuple(float(value) for value in config["betas"]),
        eps=float(config["eps"]),
    )
