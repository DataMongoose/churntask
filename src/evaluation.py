import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression


def categorical_target_summary(df, col, target='label', smoothing=0.5):
    """
    Per-category event rate, WoE, and IV contribution.
    Laplace smoothing (0.5) prevents divide-by-zero on rare categories and
    reduces IV inflation from categories with n < 10.

    WoE sign convention: ln(non_events / events) — positive means lower churn risk.
    This is the opposite of feature-engine's WoEEncoder (ln(p1/p0)), so WoE values
    here will appear sign-flipped relative to LR coefficients fitted on WoE-encoded
    features. IV is sign-invariant and unaffected.
    """
    total_ev  = df[target].sum()
    total_nev = (df[target] == 0).sum()
    g = df.groupby(col, dropna=False)[target].agg(['count', 'sum', 'mean'])
    g.columns = ['n', 'events', 'mean_target']
    g['non_events']   = g['n'] - g['events']
    g['p_events']     = (g['events']     + smoothing) / (total_ev  + smoothing * len(g))
    g['p_non_events'] = (g['non_events'] + smoothing) / (total_nev + smoothing * len(g))
    g['woe']          = np.log(g['p_non_events'] / g['p_events'])
    g['iv_contrib']   = (g['p_non_events'] - g['p_events']) * g['woe']
    return g[['n', 'events', 'mean_target', 'woe', 'iv_contrib']].sort_values(
        'mean_target', ascending=False)


def iv_band(x):
    if pd.isna(x):  return 'n/a'
    if x < 0.02:    return 'not predictive'
    if x < 0.10:    return 'weak'
    if x < 0.30:    return 'medium'
    if x < 0.50:    return 'strong'
    return 'suspicious'


def compute_iv_stability(master, cat_cols, data_end, val_months=3, target='label'):
    """
    Compute IV and Spearman ρ stability for each categorical feature
    using a train/val split on right-censoring-clean data.

    Returns (stability_df, train_c, val_c).
    """
    df_clean   = master[master['datevalue'] <= data_end].copy()
    snap_clean = sorted(df_clean['datevalue'].unique())
    val_cutoff = snap_clean[-val_months]

    train_c = df_clean[df_clean['datevalue'] <  val_cutoff].copy()
    val_c   = df_clean[df_clean['datevalue'] >= val_cutoff].copy()

    print(f'Val cutoff: {pd.Timestamp(val_cutoff).date()}')
    print(f'Train: {len(train_c):>8,} rows  churn {train_c[target].mean():.2%}')
    print(f'Val:   {len(val_c):>8,} rows  churn {val_c[target].mean():.2%}')

    rows = []
    for col in cat_cols:
        train_tbl = categorical_target_summary(train_c, col)
        val_tbl   = categorical_target_summary(val_c,   col)
        train_iv  = train_tbl['iv_contrib'].sum()
        val_iv    = val_tbl['iv_contrib'].sum()
        t_means   = train_tbl['mean_target'].rename('train')
        v_means   = val_tbl['mean_target'].rename('val')
        cmp       = pd.concat([t_means, v_means], axis=1).dropna()
        rho       = cmp.corr(method='spearman').iloc[0, 1] if len(cmp) >= 2 else np.nan
        rows.append({
            'feature':      col,
            'cardinality':  train_c[col].nunique(),
            'IV_train':     train_iv,
            'IV_val':       val_iv,
            'IV_ratio':     val_iv / train_iv if train_iv > 0 else np.nan,
            'spearman_rho': rho,
        })

    stability = (
        pd.DataFrame(rows)
        .sort_values('IV_train', ascending=False)
        .reset_index(drop=True)
    )
    stability['band_train'] = stability['IV_train'].apply(iv_band)
    stability['band_val']   = stability['IV_val'].apply(iv_band)
    return stability, train_c, val_c


def run_walk_forward_cv(master, all_features, train_window_months, data_end,
                        make_boost, make_xgb, make_lr, embargo_months=3):
    """
    Walk-forward cross-validation with 3 competing models.

    Each window: train on rolling 12 months, test on the next month.
    embargo_months (default 3): gap between last training month and test month.
    This prevents label-horizon leakage — a 90-day label window at month i-1
    overlaps heavily with the label window at month i, and the same customer
    appears in both rows with near-identical features.
    Returns a list of per-window result dicts (including raw probability arrays).
    """
    obs_months = sorted(master[master['datevalue'] <= data_end]['datevalue'].unique())
    print(f'Valid months: {obs_months[0].date()} -> {obs_months[-1].date()}  ({len(obs_months)} months)')
    print(f'Embargo: {embargo_months} month(s)  |  Expected windows: {len(obs_months) - train_window_months - embargo_months}')

    window_results = []
    for i in range(train_window_months + embargo_months, len(obs_months)):
        test_month  = obs_months[i]
        # shift training window back by embargo_months to create a purge gap
        train_slice = obs_months[i - train_window_months - embargo_months: i - embargo_months]

        df_train = master[master['datevalue'].isin(train_slice)].copy()
        df_test  = master[master['datevalue'] == test_month].copy()

        X_train, y_train = df_train[all_features], df_train['label']
        X_test,  y_test  = df_test[all_features],  df_test['label']

        if y_test.sum() == 0:
            continue

        boost      = make_boost().fit(X_train, y_train)
        boost_prob = boost.predict_proba(X_test)[:, 1]

        xgb        = make_xgb().fit(X_train, y_train)
        xgb_prob   = xgb.predict_proba(X_test)[:, 1]

        lr         = make_lr().fit(X_train, y_train)
        lr_prob    = lr.predict_proba(X_test)[:, 1]

        boost_auc = roc_auc_score(y_test, boost_prob)
        xgb_auc   = roc_auc_score(y_test, xgb_prob)
        lr_auc    = roc_auc_score(y_test, lr_prob)
        boost_pr  = average_precision_score(y_test, boost_prob)
        xgb_pr    = average_precision_score(y_test, xgb_prob)
        lr_pr     = average_precision_score(y_test, lr_prob)

        window_results.append({
            'test_month':  test_month,
            'train_start': train_slice[0],
            'train_end':   train_slice[-1],
            'n_train':     len(X_train),
            'n_test':      len(X_test),
            'n_churners':  int(y_test.sum()),
            'churn_rate':  y_test.mean(),
            'boost_auc':   boost_auc, 'boost_pr': boost_pr,
            'xgb_auc':     xgb_auc,   'xgb_pr':   xgb_pr,
            'lr_auc':      lr_auc,     'lr_pr':    lr_pr,
            'y_test':      y_test.values,
            'boost_prob':  boost_prob,
            'xgb_prob':    xgb_prob,
            'lr_prob':     lr_prob,
        })

        print(f"  {train_slice[0].strftime('%b%y')}-{train_slice[-1].strftime('%b%y')} -> "
              f"test {test_month.strftime('%b%y')}  "
              f"boost AUC={boost_auc:.3f} PR={boost_pr:.3f}  "
              f"XGB AUC={xgb_auc:.3f} PR={xgb_pr:.3f}  "
              f"LR AUC={lr_auc:.3f} PR={lr_pr:.3f}  churn={y_test.mean():.2%}")

    print(f'\nCompleted {len(window_results)} windows.')
    return window_results, obs_months


def compute_threshold_metrics(y_true, y_scores, thresholds, cost_fp, cost_fn):
    """
    Sweep thresholds and compute precision, recall, F-scores, and business cost.
    Returns a DataFrame with one row per threshold.
    """
    rows = []
    for t in thresholds:
        pred = (y_scores >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        rows.append({
            'threshold':     round(t, 4),
            'precision':     prec,
            'recall':        rec,
            'f1':            f1,
            'business_cost': cost_fp * fp + cost_fn * fn,
            'flagged_pct':   pred.mean(),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        })
    return pd.DataFrame(rows)


def calibrate_isotonic(y_true, scores):
    """
    Fit an isotonic calibrator on the first half of the data (chronologically),
    evaluate on the second half, then refit on the full set for production scoring.

    Assumes scores/y_true are ordered by time (as produced by run_walk_forward_cv).
    Returns (iso_cal, diagnostics_dict).
    """
    from sklearn.metrics import brier_score_loss

    mid = len(scores) // 2
    s_fit, y_fit = scores[:mid], y_true[:mid]
    s_val,  y_val = scores[mid:], y_true[mid:]

    iso_eval = IsotonicRegression(out_of_bounds='clip').fit(s_fit, y_fit)
    cal_val  = iso_eval.predict(s_val)

    diag = {
        'brier_raw':        brier_score_loss(y_val, s_val),
        'brier_calibrated': brier_score_loss(y_val, cal_val),
        'mean_raw':         s_val.mean(),
        'mean_calibrated':  cal_val.mean(),
        'true_rate':        y_val.mean(),
        'auc_raw':          roc_auc_score(y_val, s_val),
        'auc_calibrated':   roc_auc_score(y_val, cal_val),
    }

    iso_cal = IsotonicRegression(out_of_bounds='clip').fit(scores, y_true)
    return iso_cal, diag
