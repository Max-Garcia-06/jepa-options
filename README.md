# jepa-options

Research on JEPA-style (Joint-Embedding Predictive Architecture) self-supervised
models for options trading. Paper trading only, via Alpaca.

## Setup

```bash
uv sync
cp .env.example .env   # add Alpaca PAPER keys
uv run scripts/check_alpaca.py              # read-only account/options check
uv run scripts/check_alpaca.py --test-order # + place and cancel a non-filling paper order
```

The trading client is hardcoded to `paper=True`. This repo does not trade live.

## Status

- Alpaca paper account connectivity, options quotes, and order/cancel: verified.
- Model, data pipeline, and evaluation gate: not started.
