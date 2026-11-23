"""Pre-registered 2026-11-23. Do not edit after val results exist."""

import numpy as np

# Window geometry
H = 21
L = 60
D = 32
HIDDEN = 64

# Splits (calendar dates, inclusive); test end open = last available date
SPLITS = {
    "train": ("2008-01-02", "2016-12-30"),
    "val": ("2017-01-03", "2020-12-31"),
    "test": ("2021-01-04", None),
}
INNER = {
    "inner_fit": ("2008-01-02", "2014-12-31"),
    "inner_select": ("2015-01-02", "2016-12-30"),
}

# Costs (bps per unit turnover)
COST_VOLPT = 1.0
COST_SENS = 2.0

# Training
SEEDS = (0, 1, 2, 3, 4)
MIN_SEEDS_POSITIVE = 4
EPOCHS = 30
LR = 1e-3
WD = 1e-4
BATCH = 128
EMA = 0.996

# Loss weights
W_SIM = 25.0
W_VAR = 25.0
W_COV = 1.0

# Crash definition and clipping
CRASH_Q = 0.10
S1_CLIP = (0.0, 2.0)

# Strategy 3 grids
S3_H_GRID = (0.1, 0.2, 0.3)
S3_TAU_GRID = (0.10, 0.15, 0.20, 0.30, 0.40)

# Statistics
ALPHA = 0.05 / 3
N_BOOT = 2000
BLOCK = 21
BOOT_SEED = 12345

# Stress windows
STRESS_VAL = [("2018-01-26", "2018-02-28"), ("2020-02-19", "2020-04-30")]
STRESS_TEST = [("2022-01-03", "2022-10-14"), ("2025-04-01", "2025-04-30")]

# Probe grids
RIDGE_ALPHAS = tuple(np.logspace(-3, 3, 13))
LOGIT_CS = tuple(np.logspace(-3, 3, 13))

# Feature sets
FEATURE_SETS = {
    "core": ["ret", "rv5", "rv10", "rv21", "vix", "vix6m",
             "vvix", "skew", "vix_vix6m", "vix_rv21"],
}
FEATURE_SETS["extended"] = FEATURE_SETS["core"] + [
    "vix9d", "vix3m", "vix9d_vix", "vix3m_vix"]

# Alpaca check and financing assumptions
ALPACA_CHECK_START = "2024-02-01"
RATE = 0.04
DIV_YIELD = 0.013

# Data guards
MIN_CRASH_POSITIVES = 10
MAX_FFILL = 2
