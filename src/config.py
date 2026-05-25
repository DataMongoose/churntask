SEED = 42
PREDICTION_HORIZON_DAYS = 90
LOOKBACK_MONTHS = 3
TRAIN_WINDOW_MONTHS = 12
# Purge gap between training end and test month.
# A 90-day label horizon means month i-1's label overlaps with month i's label window,
# so we shift the training window 3 months back to prevent that leakage.
EMBARGO_MONTHS = 3
COST_FP = 6    # £ per false positive (outbound retention call; ContactBabel UK benchmark £4–£6.26)
COST_FN = 300  # £ per false negative (lost LTV; TalkTalk ARPU £25.46/mo × ~12mo retention extension)
