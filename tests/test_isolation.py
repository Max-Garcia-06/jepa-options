from pathlib import Path

import volj.train as train

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "volj/embed.py", "volj/controls.py", "volj/probes.py",
    "volj/strategies.py", "volj/gate.py", "scripts/run_round1.py",
]


def test_inference_code_never_touches_target_encoder():
    for rel in FILES:
        p = ROOT / rel
        if not p.exists():
            continue
        src = p.read_text()
        for bad in ("target_enc", "tgt_enc", "EMA"):
            assert bad not in src, f"{bad!r} found in {rel}"


def test_train_exposes_no_public_target_name():
    assert not [n for n in dir(train) if not n.startswith("_") and "target" in n.lower()]
