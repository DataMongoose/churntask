from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from feature_engine.encoding import RareLabelEncoder, MeanEncoder, WoEEncoder
from feature_engine.outliers import Winsorizer
from feature_engine.imputation import MeanMedianImputer
from feature_engine.transformation import YeoJohnsonTransformer

from src.config import SEED
from src.features import PackageGrouper

# ── Feature lists ────────────────────────────────────────────────────────────

CAT_COLS = [
    'contract_status_risk', 'technology_group', 'sales_channel_group',
    'crm_package_grouped', 'package_tech', 'package_bundle_type',
]
PKG_CAT_COLS = ['crm_package_grouped', 'package_tech', 'package_bundle_type']

NUM_COLS_AUDIT = [
    'tenure_days', 'ooc_days', 'ooc_days_capped',
    'speed', 'line_speed', 'speed_gap', 'speed_underperformance',
    'tenure_years', 'months_to_ooc',
    'calls_3m_count', 'calls_3m_loyalty', 'calls_3m_tech', 'calls_3m_csb',
    'calls_3m_avg_talk', 'calls_3m_total_talk',
    'calls_3m_avg_hold', 'calls_3m_total_hold',
    'pct_loyalty_calls',
    'calls_1m_count', 'calls_1m_loyalty', 'calls_1m_tech', 'call_recency_ratio',
    'usage_3m_avg_download', 'usage_3m_avg_upload',
    'usage_3m_total_download', 'usage_3m_total_upload',
    'usage_3m_days', 'download_trend_mom',
    'usage_3m_download_std', 'usage_3m_upload_std',
    'download_pct_change', 'download_volatility_cv',
]
BIN_COLS_AUDIT = [
    'contract_dd_cancels', 'dd_cancel_60_day',
    'has_dd_cancel', 'is_ooc', 'has_loyalty_call',
    'is_legacy_tech', 'is_digital_acquisition', 'is_migrated',
    'recent_loyalty_call',
]

# Manual drop of known redundant/collinear features
DROPPED_COLLINEAR = [
    'months_to_ooc',          # = -ooc_days / 30.44      (exact linear transform)
    'tenure_years',           # = tenure_days / 365.25   (exact linear transform)
    'ooc_days_capped',        # = ooc_days.clip(0, 365)  (redundant variant)
    'has_dd_cancel',          # = (contract_dd_cancels > 0) (redundant flag)
    'speed_underperformance', # = -(speed_gap)/speed      (redundant variant)
    'package_speed',          # corr(speed) > 0.95; speed_gap already captures it
]

NUM_COLS = [c for c in NUM_COLS_AUDIT if c not in DROPPED_COLLINEAR]
BIN_COLS = [c for c in BIN_COLS_AUDIT if c not in DROPPED_COLLINEAR]

# Winsorizer: skip zero-inflated counts where IQR=0 breaks capping
WINSOR_EXCLUDE = {
    'calls_3m_count', 'calls_3m_loyalty', 'calls_3m_tech', 'pct_loyalty_calls',
    'calls_1m_count', 'calls_1m_loyalty', 'calls_1m_tech', 'call_recency_ratio',
}
WINSOR_COLS = [c for c in NUM_COLS if c not in WINSOR_EXCLUDE]

# Pipeline input: PackageGrouper takes crm_package_name and expands it inside the pipeline
ALL_FEATURES = (
    ['crm_package_name'] +
    [c for c in CAT_COLS if c not in PKG_CAT_COLS] +
    NUM_COLS +
    BIN_COLS
)

# Audit set: all features before the collinearity drop (used in diagnostic)
ALL_FEATURES_AUDIT = CAT_COLS + NUM_COLS_AUDIT + BIN_COLS_AUDIT
CAT_COLS_AUDIT = CAT_COLS


# ── Pipeline factories ───────────────────────────────────────────────────────

def make_boost_pipeline():
    """PackageGrouper → RareLabelEncoder → MeanEncoder → Winsorizer → HistGradientBoostingClassifier"""
    return Pipeline([
        ('pkg',    PackageGrouper()),
        ('rare',   RareLabelEncoder(
                       tol=0.02, n_categories=5,
                       variables=CAT_COLS, missing_values='ignore')),
        ('encode', MeanEncoder(
                       variables=CAT_COLS, missing_values='ignore')),
        ('winsor', Winsorizer(
                       capping_method='iqr', tail='both', fold=3.0,
                       variables=WINSOR_COLS, missing_values='ignore')),
        ('model',  HistGradientBoostingClassifier(
                       max_iter=300, learning_rate=0.05,
                       max_leaf_nodes=63, min_samples_leaf=50,
                       class_weight='balanced', random_state=SEED,
                       early_stopping=True, validation_fraction=0.1,
                       n_iter_no_change=20, verbose=0)),
    ])


def make_xgb_pipeline():
    """PackageGrouper → RareLabelEncoder → MeanEncoder → Winsorizer → XGBClassifier"""
    # scale_pos_weight=7 mirrors class_weight='balanced' for XGBoost at ~12.6% churn rate
    return Pipeline([
        ('pkg',    PackageGrouper()),
        ('rare',   RareLabelEncoder(
                       tol=0.02, n_categories=5,
                       variables=CAT_COLS, missing_values='ignore')),
        ('encode', MeanEncoder(
                       variables=CAT_COLS, missing_values='ignore')),
        ('winsor', Winsorizer(
                       capping_method='iqr', tail='both', fold=3.0,
                       variables=WINSOR_COLS, missing_values='ignore')),
        ('model',  XGBClassifier(
                       n_estimators=300, learning_rate=0.05,
                       max_depth=5, min_child_weight=50,
                       scale_pos_weight=7,
                       random_state=SEED,
                       eval_metric='logloss', verbosity=0)),
    ])


def make_lr_pipeline():
    """PackageGrouper → RareLabelEncoder → WoEEncoder → MeanMedianImputer → YeoJohnson → StandardScaler → LR"""
    return Pipeline([
        ('pkg',    PackageGrouper()),
        ('rare',   RareLabelEncoder(
                       tol=0.02, n_categories=5,
                       variables=CAT_COLS, missing_values='ignore')),
        ('woe',    WoEEncoder(variables=CAT_COLS)),
        ('impute', MeanMedianImputer(
                       imputation_method='median', variables=NUM_COLS)),
        ('yj',     YeoJohnsonTransformer(variables=NUM_COLS)),
        ('scale',  StandardScaler()),
        ('model',  LogisticRegression(
                       class_weight='balanced', max_iter=1000,
                       C=0.1, solver='lbfgs', random_state=SEED)),
    ])
