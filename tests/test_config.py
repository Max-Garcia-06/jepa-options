from volj import config as c

def test_preregistered_values():
    assert (c.H, c.L, c.D, c.HIDDEN) == (21, 60, 32, 64)
    assert c.SPLITS["train"] == ("2008-01-02", "2016-12-30")
    assert c.SPLITS["val"] == ("2017-01-03", "2020-12-31")
    assert c.SPLITS["test"] == ("2021-01-04", None)
    assert c.INNER["inner_fit"] == ("2008-01-02", "2014-12-31")
    assert c.INNER["inner_select"] == ("2015-01-02", "2016-12-30")
    assert c.COST_VOLPT == 1.0 and c.COST_SENS == 2.0
    assert c.SEEDS == (0, 1, 2, 3, 4) and c.MIN_SEEDS_POSITIVE == 4
    assert (c.EPOCHS, c.LR, c.WD, c.BATCH, c.EMA) == (30, 1e-3, 1e-4, 128, 0.996)
    assert (c.W_SIM, c.W_VAR, c.W_COV) == (25.0, 25.0, 1.0)
    assert c.CRASH_Q == 0.10 and c.S1_CLIP == (0.0, 2.0)
    assert c.S3_H_GRID == (0.1, 0.2, 0.3)
    assert c.S3_TAU_GRID == (0.10, 0.15, 0.20, 0.30, 0.40)
    assert abs(c.ALPHA - 0.05 / 3) < 1e-12
    assert (c.N_BOOT, c.BLOCK, c.BOOT_SEED) == (2000, 21, 12345)
    assert c.STRESS_VAL == [("2018-01-26", "2018-02-28"), ("2020-02-19", "2020-04-30")]
    assert c.STRESS_TEST == [("2022-01-03", "2022-10-14"), ("2025-04-01", "2025-04-30")]
    assert len(c.RIDGE_ALPHAS) == 13 and len(c.LOGIT_CS) == 13
    assert c.FEATURE_SETS["core"] == ["ret", "rv5", "rv10", "rv21", "vix", "vix6m",
                                      "vvix", "skew", "vix_vix6m", "vix_rv21"]
    assert c.FEATURE_SETS["extended"] == c.FEATURE_SETS["core"] + [
        "vix9d", "vix3m", "vix9d_vix", "vix3m_vix"]
    assert (c.ALPACA_CHECK_START, c.RATE, c.DIV_YIELD) == ("2024-02-01", 0.04, 0.013)
    assert c.MIN_CRASH_POSITIVES == 10 and c.MAX_FFILL == 2
