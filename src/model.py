import math
from typing import List

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        attn_out, _ = self.self_attn(
            y,
            y,
            y,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(attn_out)

        y = self.norm2(x)
        x = x + self.drop2(self.ff(y))
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        y = self.norm1(x)
        self_attn_out, _ = self.self_attn(
            y,
            y,
            y,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(self_attn_out)

        y = self.norm2(x)
        cross_attn_out, _ = self.cross_attn(
            y,
            memory,
            memory,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop2(cross_attn_out)

        y = self.norm3(x)
        x = x + self.drop3(self.ff(y))
        return x


class Seq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        tgt_sos_idx: int,
        dropout: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.d_model = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.embed_dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.enc_final_norm = nn.LayerNorm(d_model)
        self.dec_final_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_key_padding_mask = src == self.src_pad_idx
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.embed_dropout(self.pos_enc(x))
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.enc_final_norm(x)
        return x, src_key_padding_mask

    def decode(self, tgt_in: torch.Tensor, memory: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        tgt_key_padding_mask = tgt_in == self.tgt_pad_idx
        tgt_mask = self._causal_mask(tgt_in.size(1), tgt_in.device)

        x = self.tgt_embed(tgt_in) * math.sqrt(self.d_model)
        x = self.embed_dropout(self.pos_enc(x))
        for layer in self.decoder_layers:
            x = layer(
                x,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        x = self.dec_final_norm(x)
        return self.out_proj(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing: float = 1.0) -> torch.Tensor:
        del teacher_forcing
        memory, src_key_padding_mask = self.encode(src)
        tgt_in = tgt[:, :-1]
        return self.decode(tgt_in, memory, src_key_padding_mask)

    @torch.no_grad()
    def translate(
        self, src_ids: List[int], max_len: int, device: torch.device, eos_idx: int
    ) -> List[int]:
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        memory, src_key_padding_mask = self.encode(src)

        out_ids: List[int] = [self.tgt_sos_idx]
        for _ in range(max_len):
            tgt_in = torch.tensor([out_ids], dtype=torch.long, device=device)
            logits = self.decode(tgt_in, memory, src_key_padding_mask)
            next_token = int(logits[:, -1, :].argmax(dim=-1).item())
            out_ids.append(next_token)
            if next_token == eos_idx:
                break
        return out_ids
