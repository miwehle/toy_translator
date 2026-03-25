# Minimaler Transformer-Translator (PyTorch)

Dieses Projekt enthaelt einen bewusst einfachen Transformer-Translator mit:

- Transformer-Encoder mit Multi-Head Self-Attention
- Transformer-Decoder mit Masked Self-Attention und Cross-Attention
- Pre-Norm (LayerNorm vor jedem Sub-Block)
- Feed-Forward-Netzwerken pro Block

## Struktur

- `simple_attention_translator.py` (Entry-Point)
- `src/constants.py` (Sondertokens)
- `src/data.py` (Vokabular, Dataset, Collate, Toy-Korpus)
- `src/model.py` (Transformer-Modell)
- `src/train.py` (Training, CLI, Beispiel-Inferenz)

## Start

```bash
python simple_attention_translator.py --epochs 120
```

## Checkpoint und Inferenz

Training speichert automatisch ein Checkpoint unter `checkpoints/translator.pt` (anpassbar mit `--checkpoint-path`).

Einzelsatz uebersetzen:

```bash
python simple_attention_translator.py --checkpoint-path checkpoints/translator.pt --translate "ich bin muede"
```

Interaktiv testen:

```bash
python simple_attention_translator.py --checkpoint-path checkpoints/translator.pt --interactive
```

## Wichtige Parameter

- `--epochs` (default: `200`)
- `--batch-size` (default: `4`)
- `--emb-dim` (default: `64`)
- `--hidden-dim` (default: `64`, Feed-Forward-Dimension)
- `--lr` (default: `1e-3`)
- `--num-heads` (default: `4`)
- `--num-layers` (default: `2`)
- `--dropout` (default: `0.1`)
- `--checkpoint-path` (default: `checkpoints/translator.pt`)
- `--translate` (default: `None`)
- `--interactive` (Flag)
- `--max-len` (default: `30`, nur Inferenz)

## Hinweis

Der Datensatz ist absichtlich sehr klein (Toy-Parallelkorpus in der Datei), damit das Beispiel schnell verstaendlich und trainierbar bleibt.
