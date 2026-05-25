import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ── Contract status ──────────────────────────────────────────────────────────

CONTRACT_STATUS_VALID = {
    '01 Early Contract', '02 In Contract', '03 Soon to be OOC',
    '04 Coming OOC', '05 Newly OOC', '06 OOC',
}
CONTRACT_STATUS_RISK_MAP = {
    '01 Early Contract':  'in_contract',
    '02 In Contract':     'in_contract',
    '03 Soon to be OOC':  'approaching_ooc',
    '04 Coming OOC':      'approaching_ooc',
    '05 Newly OOC':       'ooc',
    '06 OOC':             'ooc',
}


def transform_contract_status(df, col='contract_status'):
    df = df.copy()
    df['contract_status_clean'] = df[col].where(
        df[col].isin(CONTRACT_STATUS_VALID), other='unknown').fillna('unknown')
    df['contract_status_risk'] = df['contract_status_clean'].map(
        CONTRACT_STATUS_RISK_MAP).fillna('unknown')
    return df


# ── Technology ───────────────────────────────────────────────────────────────

TECHNOLOGY_GROUP_MAP = {
    'MPF':     'legacy',
    'FTTC':    'standard_fibre',
    'GFAST':   'next_gen',
    'FTTP':    'next_gen',
    'Missing': 'unknown',
}


def transform_technology(df, col='technology'):
    df = df.copy()
    df['technology_clean'] = df[col].fillna('Missing')
    df['technology_group'] = df['technology_clean'].map(TECHNOLOGY_GROUP_MAP).fillna('unknown')
    df['is_legacy_tech']   = (df['technology_clean'] == 'MPF').astype(np.int8)
    return df


# ── Sales channel ────────────────────────────────────────────────────────────

SALES_CHANNEL_GROUP_MAP = {
    'Online - Affiliate':  'Online',
    'Online - Search':     'Online',
    'Online - Ambient':    'Online',
    'Online - Other':      'Other',
    'Migrated Customer':   'Migrated_Customer',
    'Inbound':             'Inbound',
    'Unknown':             'Unknown',
    'Partners':            'Partners',
    'Retail':              'Retail',
    'Webchat':             'Webchat',
    'Field':               'Other',
    'Outbound':            'Other',
    'Other':               'Other',
}


def transform_sales_channel(df, col='sales_channel'):
    df = df.copy()
    df['sales_channel_group']     = df[col].map(SALES_CHANNEL_GROUP_MAP).fillna('Unknown')
    df['is_digital_acquisition']  = (df['sales_channel_group'] == 'Online').astype(np.int8)
    df['is_migrated']             = (df['sales_channel_group'] == 'Migrated_Customer').astype(np.int8)
    return df


# ── CRM package name — structured parse ─────────────────────────────────────

_SPEED_PATTERN = re.compile(r'\b(35|65|150|250|500|900)\b')
_TECH_RULES = [
    ('fttp_cf', re.compile(r'FTTP-CF', re.I)),
    ('fttp_or', re.compile(r'FTTP-OR', re.I)),
    ('gfast',   re.compile(r'GFAST',   re.I)),
    ('fttc',    re.compile(r'FTTC',    re.I)),
    ('mpf',     re.compile(r'\bS?MPF\b', re.I)),
    ('ips',     re.compile(r'\bIPS\b', re.I)),
    ('wlr',     re.compile(r'\bWLR\b', re.I)),
]
_BUNDLE_RULES = [
    ('broadband_tv',    re.compile(r'\bTV\b',            re.I)),
    ('broadband_voip',  re.compile(r'\bVOIP\b',          re.I)),
    ('voice_only',      re.compile(r'voice only',        re.I)),
    ('broadband_only',  re.compile(r'data only',         re.I)),
    ('broadband_calls', re.compile(r'calls?|bb with',    re.I)),
    ('broadband_only',  re.compile(r'broadband|fibre|\bBB\b', re.I)),
]


def _parse_speed(name):
    if not isinstance(name, str): return np.nan
    m = _SPEED_PATTERN.search(name)
    return float(m.group(1)) if m else np.nan


def _parse_tech(name):
    if not isinstance(name, str): return 'unknown'
    for label, pattern in _TECH_RULES:
        if pattern.search(name): return label
    return 'unknown'


def _parse_bundle(name):
    if not isinstance(name, str): return 'unknown'
    for label, pattern in _BUNDLE_RULES:
        if pattern.search(name): return label
    return 'other'


class PackageGrouper(BaseEstimator, TransformerMixin):
    """
    Parses crm_package_name into structured features and groups rare packages.
    Top-N is frequency-only (no target). Embedded as step 0 of every pipeline
    so it is always refit on the training slice passed to pipeline.fit().
    Drops crm_package_name from the output; downstream steps see only the
    three derived columns: crm_package_grouped, package_tech, package_bundle_type.
    """

    def __init__(self, col='crm_package_name', top_n=10):
        self.col = col
        self.top_n = top_n
        self._top_packages = set()

    def fit(self, X, y=None):
        self._top_packages = set(X[self.col].value_counts().nlargest(self.top_n).index)
        return self

    def transform(self, X):
        df = X.copy()
        df['package_tech']        = df[self.col].apply(_parse_tech)
        df['package_bundle_type'] = df[self.col].apply(_parse_bundle)
        df['crm_package_grouped'] = df[self.col].apply(
            lambda x: x if x in self._top_packages else 'Other_Package')
        return df.drop(columns=[self.col])


PKG_CAT_COLS = ['crm_package_grouped', 'package_tech', 'package_bundle_type']


def transform_all_categorical(df):
    """Apply all stateless categorical transforms (safe to run on the full dataset)."""
    df = transform_contract_status(df)
    df = transform_technology(df)
    df = transform_sales_channel(df)
    return df


# ── Call features ────────────────────────────────────────────────────────────

def compute_call_features(calls, lookback_months=3):
    """
    Aggregate raw call records into 3-month call features per (customer, datevalue).

    Returns (call_feats, calls_monthly):
      - call_feats: one row per (customer, datevalue) — the feature table
      - calls_monthly: intermediate monthly aggregates (needed for temporal features)
    """
    calls = calls.copy()
    calls['call_month'] = calls['event_date'].dt.to_period('M').dt.to_timestamp()

    major_types = ['Tech', 'CS&B', 'Loyalty', 'Customer Finance', 'FTTP']
    calls['call_type'] = calls['call_type'].astype(str).apply(lambda x: x.lower())
    major_types_lower  = [x.lower() for x in major_types]
    calls['call_type_grouped'] = np.where(
        calls['call_type'].isin(major_types_lower), calls['call_type'], 'Other')

    call_dummies = pd.get_dummies(calls['call_type_grouped'], prefix='call').astype(np.int8)
    call_dummies.columns = [c.replace(' ', '_').replace('&', '_') for c in call_dummies.columns]
    calls    = pd.concat([calls, call_dummies], axis=1)
    dummy_cols = call_dummies.columns.tolist()

    calls_monthly = (
        calls.groupby(['unique_customer_identifier', 'call_month'])
        .agg(
            call_count     =('event_date',       'count'),
            total_talk_sec =('talk_time_seconds', 'sum'),
            total_hold_sec =('hold_time_seconds', 'sum'),
            **{f'{c}_count': (c, 'sum') for c in dummy_cols},
        ).reset_index()
    )
    print(f'calls_monthly: {calls_monthly.shape}')

    expanded = []
    for offset in range(1, lookback_months + 1):
        tmp = calls_monthly.copy()
        tmp['datevalue'] = tmp['call_month'] + pd.DateOffset(months=offset)
        expanded.append(tmp)

    call_feats = (
        pd.concat(expanded, ignore_index=True)
        .groupby(['unique_customer_identifier', 'datevalue'])
        .agg(
            calls_3m_count      =('call_count',     'sum'),
            calls_3m_total_talk =('total_talk_sec', 'sum'),
            calls_3m_total_hold =('total_hold_sec', 'sum'),
            **{f'{c}_3m': (f'{c}_count', 'sum') for c in dummy_cols},
        ).reset_index()
    )

    call_feats['calls_3m_avg_talk'] = call_feats['calls_3m_total_talk'] / call_feats['calls_3m_count']
    call_feats['calls_3m_avg_hold'] = call_feats['calls_3m_total_hold'] / call_feats['calls_3m_count']

    for c in dummy_cols:
        call_feats[f'{c}_3m_ratio'] = call_feats[f'{c}_3m'] / call_feats['calls_3m_count']

    call_feats['datevalue'] = pd.to_datetime(call_feats['datevalue'])
    call_feats = call_feats.replace([np.inf, -np.inf], np.nan)

    # Stable aliases used by downstream feature lists
    call_feats['calls_3m_loyalty'] = call_feats['call_loyalty_3m']
    call_feats['calls_3m_tech']    = call_feats['call_tech_3m']
    call_feats['calls_3m_csb']     = call_feats['call_cs_b_3m']

    print(f'call_feats: {call_feats.shape}')
    return call_feats, calls_monthly


# ── Usage features ───────────────────────────────────────────────────────────

def compute_usage_features(usage_monthly, lookback_months=3):
    """
    Forward-assign monthly usage data into 3-month rolling features per (customer, datevalue).

    Returns (usage_feats, usage_expanded):
      - usage_feats: one row per (customer, datevalue) — the feature table
      - usage_expanded: list of per-offset dataframes (needed for compute_temporal_features)
    """
    usage_expanded = []
    for offset in range(1, lookback_months + 1):
        tmp = usage_monthly.copy()
        tmp['datevalue'] = tmp['year_month'] + pd.DateOffset(months=offset)
        usage_expanded.append(tmp)

    usage_feats_3m = (
        pd.concat(usage_expanded, ignore_index=True)
        .groupby(['unique_customer_identifier', 'datevalue'])
        .agg(
            usage_3m_total_download=('total_download_mbs', 'sum'),
            usage_3m_total_upload  =('total_upload_mbs',   'sum'),
            usage_3m_days          =('usage_days_recorded','sum'),
        ).reset_index()
    )
    usage_feats_3m['usage_3m_avg_download'] = (
        usage_feats_3m['usage_3m_total_download'] / usage_feats_3m['usage_3m_days'])
    usage_feats_3m['usage_3m_avg_upload'] = (
        usage_feats_3m['usage_3m_total_upload'] / usage_feats_3m['usage_3m_days'])
    usage_feats_3m['datevalue'] = pd.to_datetime(usage_feats_3m['datevalue'])

    # Month-over-month download trend (most recent month vs the one before)
    u_last = usage_monthly.copy()
    u_last['datevalue'] = u_last['year_month'] + pd.DateOffset(months=1)
    u_last = u_last[['unique_customer_identifier', 'datevalue', 'avg_download_mbs']].rename(
        columns={'avg_download_mbs': 'dl_last_m'})

    u_prev = usage_monthly.copy()
    u_prev['datevalue'] = u_prev['year_month'] + pd.DateOffset(months=2)
    u_prev = u_prev[['unique_customer_identifier', 'datevalue', 'avg_download_mbs']].rename(
        columns={'avg_download_mbs': 'dl_prev_m'})

    trend = u_last.merge(u_prev, on=['unique_customer_identifier', 'datevalue'], how='outer')
    trend['download_trend_mom'] = trend['dl_last_m'] - trend['dl_prev_m']
    trend['datevalue'] = pd.to_datetime(trend['datevalue'])

    usage_feats = usage_feats_3m.merge(
        trend[['unique_customer_identifier', 'datevalue', 'download_trend_mom']],
        on=['unique_customer_identifier', 'datevalue'], how='left',
    )
    usage_feats = usage_feats.replace([np.inf, -np.inf], np.nan)
    usage_feats['datevalue'] = pd.to_datetime(usage_feats['datevalue'])

    print(f'usage_feats: {usage_feats.shape}')
    return usage_feats, usage_expanded


# ── Temporal features ────────────────────────────────────────────────────────

def compute_temporal_features(calls_monthly, usage_expanded):
    """
    Compute recency and volatility signals.

    Returns (calls_1m, usage_volatility):
      - calls_1m: last-month call counts per (customer, datevalue)
      - usage_volatility: std of monthly downloads across the 3-month window
    """
    calls_last = calls_monthly.copy()
    calls_last['datevalue'] = calls_last['call_month'] + pd.DateOffset(months=1)
    calls_1m = (
        calls_last
        .rename(columns={
            'call_count':         'calls_1m_count',
            'call_loyalty_count': 'calls_1m_loyalty',
            'call_tech_count':    'calls_1m_tech',
        })
        [['unique_customer_identifier', 'datevalue',
          'calls_1m_count', 'calls_1m_loyalty', 'calls_1m_tech']]
    )
    calls_1m['datevalue'] = pd.to_datetime(calls_1m['datevalue'])
    print(f'calls_1m: {calls_1m.shape}')

    usage_volatility = (
        pd.concat(usage_expanded, ignore_index=True)
        .groupby(['unique_customer_identifier', 'datevalue'])
        .agg(
            usage_3m_download_std=('avg_download_mbs', 'std'),
            usage_3m_upload_std  =('avg_upload_mbs',   'std'),
        ).reset_index()
    )
    usage_volatility['datevalue'] = pd.to_datetime(usage_volatility['datevalue'])
    print(f'usage_volatility: {usage_volatility.shape}')
    return calls_1m, usage_volatility


# ── Master table assembly ────────────────────────────────────────────────────

def build_master(df, call_feats, usage_feats, calls_1m, usage_volatility):
    """Left-join all feature tables onto the (customer, datevalue) spine."""
    master = df.merge(call_feats,       on=['unique_customer_identifier', 'datevalue'], how='left')
    master = master.merge(usage_feats,  on=['unique_customer_identifier', 'datevalue'], how='left')
    master = master.merge(calls_1m,     on=['unique_customer_identifier', 'datevalue'], how='left')
    master = master.merge(usage_volatility, on=['unique_customer_identifier', 'datevalue'], how='left')
    assert len(master) == len(df), 'A feature merge fanned out the grain!'
    print(f'master: {master.shape}  |  label rate: {master["label"].mean():.3%}')
    return master


def add_derived_features(master):
    """
    Compute derived numeric features, fill NaNs, apply stateless categorical transforms,
    and add package-parsed columns for EDA/IV analysis (without running PackageGrouper).
    """
    master = master.copy()

    # Speed signals
    master['speed_gap']             = master['line_speed'] - master['speed']
    master['speed_ratio']           = master['line_speed'] / master['speed'].replace(0, np.nan)
    master['speed_underperformance'] = (
        (master['speed'] - master['line_speed']) / master['speed'].replace(0, np.nan)
    ).clip(-1, 1)

    # OOC flags
    master['is_ooc']          = (master['ooc_days'] > 0).astype(np.int8)
    master['ooc_days_capped'] = master['ooc_days'].clip(0, 365)
    master['has_dd_cancel']   = (master['contract_dd_cancels'] > 0).astype(np.int8)

    # 3-month call derivatives
    master['calls_3m_count']      = master['calls_3m_count'].fillna(0)
    master['calls_3m_loyalty']    = master['calls_3m_loyalty'].fillna(0)
    master['calls_3m_tech']       = master['calls_3m_tech'].fillna(0)
    # Customers with zero calls have NaN talk/hold from the division in compute_call_features;
    # fill to 0 so the LR pipeline's median-imputer doesn't assign a positive duration to them.
    master['calls_3m_total_talk'] = master['calls_3m_total_talk'].fillna(0)
    master['calls_3m_total_hold'] = master['calls_3m_total_hold'].fillna(0)
    master['calls_3m_avg_talk']   = master['calls_3m_avg_talk'].fillna(0)
    master['calls_3m_avg_hold']   = master['calls_3m_avg_hold'].fillna(0)
    master['pct_loyalty_calls'] = np.where(
        master['calls_3m_count'] > 0,
        master['calls_3m_loyalty'] / master['calls_3m_count'], 0.0)
    master['has_loyalty_call']  = (master['calls_3m_loyalty'] > 0).astype(np.int8)

    # Temporal / recency
    master['calls_1m_count']     = master['calls_1m_count'].fillna(0)
    master['calls_1m_loyalty']   = master['calls_1m_loyalty'].fillna(0)
    master['calls_1m_tech']      = master['calls_1m_tech'].fillna(0)
    master['call_recency_ratio'] = master['calls_1m_count'] / (master['calls_3m_count'] / 3 + 0.5)
    master['recent_loyalty_call'] = (master['calls_1m_loyalty'] > 0).astype(np.int8)

    # Interpretability rescales (EDA only — excluded from model features)
    master['tenure_years']  = master['tenure_days'] / 365.25
    master['months_to_ooc'] = -master['ooc_days'] / 30.44
    master['download_pct_change'] = master['download_trend_mom'] / (
        master['usage_3m_avg_download'].replace(0, np.nan) + 1)
    master['download_volatility_cv'] = master['usage_3m_download_std'] / (
        master['usage_3m_avg_download'] + 1)

    # Stateless categorical transforms
    for col in ['contract_status', 'technology', 'sales_channel', 'crm_package_name']:
        master[col] = master[col].fillna('Missing')
    master = transform_all_categorical(master)
    for col in ['contract_status_risk', 'technology_group', 'sales_channel_group']:
        master[col] = master[col].fillna('unknown')

    # Package columns for EDA/IV (global, stateless — PackageGrouper is refit per fold in pipelines)
    master['package_tech']        = master['crm_package_name'].apply(_parse_tech)
    master['package_bundle_type'] = master['crm_package_name'].apply(_parse_bundle)
    _top_pkgs = set(master['crm_package_name'].value_counts().nlargest(10).index)
    master['crm_package_grouped'] = master['crm_package_name'].apply(
        lambda x: x if x in _top_pkgs else 'Other_Package')

    print(f'master final shape: {master.shape}')
    return master
