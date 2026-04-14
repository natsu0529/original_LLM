from __future__ import annotations

import argparse
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_head
        self.context_length = config.context_length

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context_length {self.context_length}"
            )

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        y = self.out_proj(y)
        return self.resid_dropout(y)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_hidden),
            nn.GELU(),
            nn.Linear(config.ffn_hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if idx.ndim != 2:
            raise ValueError(f"Expected idx shape [B, T], got {tuple(idx.shape)}")
        if idx.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"Expected integer token ids, got {idx.dtype}")

        batch_size, seq_len = idx.shape
        if seq_len > self.config.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context_length {self.config.context_length}"
            )
        if targets is not None and targets.shape != idx.shape:
            raise ValueError(
                f"Expected targets shape {tuple(idx.shape)}, got {tuple(targets.shape)}"
            )

        positions = torch.arange(seq_len, device=idx.device)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(positions).unsqueeze(0)

        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, self.config.vocab_size),
                targets.reshape(batch_size * seq_len),
            )
        return logits, loss


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal self-check for the decoder-only model.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    config = ModelConfig(context_length=args.context_length, dropout=args.dropout)
    model = DecoderOnlyTransformer(config)

    x = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(args.batch_size, args.context_length),
        dtype=torch.long,
    )
    logits, loss = model(x, x)

    print(f"model_config={asdict(config)}")
    print(f"parameter_count={count_parameters(model)}")
    print(f"input_shape={tuple(x.shape)}")
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"loss={loss.item():.6f}")

    too_long = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, config.context_length + 1),
        dtype=torch.long,
    )
    try:
        model(too_long)
    except ValueError as exc:
        print(f"context_overflow_check=ok message={exc}")
    else:
        raise AssertionError("Expected context length overflow to raise ValueError")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
