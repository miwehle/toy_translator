import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from .constants import EOS, PAD, SOS, UNK


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def tokenize(text: str) -> List[str]:
    return text.lower().strip().split()


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(cls, sentences: List[str]) -> "Vocab":
        tokens = [PAD, SOS, EOS, UNK]
        seen = set(tokens)
        for sent in sentences:
            for tok in tokenize(sent):
                if tok not in seen:
                    seen.add(tok)
                    tokens.append(tok)
        stoi = {tok: i for i, tok in enumerate(tokens)}
        return cls(stoi=stoi, itos=tokens)

    def encode(self, sentence: str) -> List[int]:
        ids = [self.stoi[SOS]]
        ids.extend(self.stoi.get(tok, self.stoi[UNK]) for tok in tokenize(sentence))
        ids.append(self.stoi[EOS])
        return ids

    def decode(self, ids: List[int]) -> str:
        words: List[str] = []
        for idx in ids:
            tok = self.itos[idx]
            if tok in {SOS, PAD}:
                continue
            if tok == EOS:
                break
            words.append(tok)
        return " ".join(words)

    @property
    def size(self) -> int:
        return len(self.itos)


class TranslationDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], src_vocab: Vocab, tgt_vocab: Vocab):
        self.data = [(src_vocab.encode(src), tgt_vocab.encode(tgt)) for src, tgt in pairs]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.data[idx]


def collate_fn(
    batch: List[Tuple[List[int], List[int]]], pad_idx_src: int, pad_idx_tgt: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_src = max(len(src) for src, _ in batch)
    max_tgt = max(len(tgt) for _, tgt in batch)
    bsz = len(batch)

    src_batch = torch.full((bsz, max_src), pad_idx_src, dtype=torch.long)
    tgt_batch = torch.full((bsz, max_tgt), pad_idx_tgt, dtype=torch.long)

    for i, (src, tgt) in enumerate(batch):
        src_batch[i, : len(src)] = torch.tensor(src, dtype=torch.long)
        tgt_batch[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)
    return src_batch, tgt_batch


def tiny_parallel_corpus() -> List[Tuple[str, str]]:
    return [
        ("ich bin muede", "i am tired"),
        ("ich bin hungrig", "i am hungry"),
        ("ich liebe kaffee", "i love coffee"),
        ("ich trinke wasser", "i drink water"),
        ("guten morgen", "good morning"),
        ("gute nacht", "good night"),
        ("wie geht es dir", "how are you"),
        ("mir geht es gut", "i am fine"),
        ("danke", "thank you"),
        ("bis spaeter", "see you later"),
        ("wo ist der bahnhof", "where is the train station"),
        ("ich lerne deutsch", "i am learning german"),
    ]
