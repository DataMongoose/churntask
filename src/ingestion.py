import os
import numpy as np
import pandas as pd
import duckdb


def read_and_parse_data(file_name, id_col='unique_customer_identifier', parse_dates=None):
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_name, parse_dates=parse_dates)
    else:
        df = pd.read_parquet(file_name, engine='pyarrow')
        if parse_dates:
            for c in parse_dates:
                df[c] = pd.to_datetime(df[c])
    print(f'{file_name}: {df.shape}  |  {df[id_col].nunique():,} customers')
    return df


def validate_ids(df, id_col, valid_ids, df_name):
    mask = df[id_col].isin(valid_ids)
    dropped = df.loc[~mask, id_col].nunique()
    if dropped:
        print(f'  {df_name}: dropped {dropped} IDs not in customer_info')
    return df[mask]


def load_raw_data(data_dir='.'):
    """Load customer_info, calls, cease, and aggregate usage monthly via DuckDB."""
    customer_info = read_and_parse_data(
        os.path.join(data_dir, 'customer_info.parquet'), parse_dates=['datevalue'])
    calls = read_and_parse_data(
        os.path.join(data_dir, 'calls.csv'), parse_dates=['event_date'])
    cease = read_and_parse_data(
        os.path.join(data_dir, 'cease.csv'),
        parse_dates=['cease_placed_date', 'cease_completed_date'])

    print('Aggregating usage via DuckDB (83M rows)...')
    usage_path = os.path.join(data_dir, 'usage.parquet')
    con = duckdb.connect()
    usage_monthly = con.execute(f"""
        SELECT
            unique_customer_identifier,
            DATE_TRUNC('month', CAST(calendar_date AS DATE)) AS year_month,
            AVG(TRY_CAST(usage_download_mbs AS DOUBLE)) AS avg_download_mbs,
            AVG(TRY_CAST(usage_upload_mbs   AS DOUBLE)) AS avg_upload_mbs,
            SUM(TRY_CAST(usage_download_mbs AS DOUBLE)) AS total_download_mbs,
            SUM(TRY_CAST(usage_upload_mbs   AS DOUBLE)) AS total_upload_mbs,
            COUNT(*) AS usage_days_recorded
        FROM read_parquet('{usage_path}')
        GROUP BY 1, 2
    """).df()
    usage_monthly['year_month'] = pd.to_datetime(usage_monthly['year_month'])
    print(f'usage_monthly: {usage_monthly.shape}')

    valid_ids = customer_info['unique_customer_identifier']
    calls         = validate_ids(calls,         'unique_customer_identifier', valid_ids, 'calls')
    cease         = validate_ids(cease,         'unique_customer_identifier', valid_ids, 'cease')
    usage_monthly = validate_ids(usage_monthly, 'unique_customer_identifier', valid_ids, 'usage_monthly')

    return customer_info, calls, cease, usage_monthly


def build_labels(customer_info, cease, prediction_horizon_days=90):
    """
    Merge first cease date, remove already-churned rows, build binary label,
    and derive the right-censoring guard DATA_END from the data itself.

    Returns (df, DATA_END).
    """
    first_cease = (
        cease.groupby('unique_customer_identifier')['cease_placed_date']
        .min().reset_index()
        .rename(columns={'cease_placed_date': 'first_cease_date'})
    )

    df = customer_info.merge(first_cease, on='unique_customer_identifier', how='left')
    n_before = len(df)
    df = df[df['first_cease_date'].isna() | (df['first_cease_date'] > df['datevalue'])].copy()
    print(f'Removed {n_before - len(df):,} already-churned rows. Remaining: {len(df):,}')

    df['label'] = (
        (df['first_cease_date'] > df['datevalue']) &
        (df['first_cease_date'] <= df['datevalue'] + pd.Timedelta(days=prediction_horizon_days))
    ).astype(np.int8)

    MAX_CEASE_DATE = cease['cease_placed_date'].max()
    DATA_END = MAX_CEASE_DATE - pd.Timedelta(days=prediction_horizon_days)
    DATA_END = DATA_END.replace(day=1) - pd.DateOffset(months=1)
    buffer_days = (MAX_CEASE_DATE - (DATA_END + pd.Timedelta(days=prediction_horizon_days))).days

    print(f'Max cease date in data  : {MAX_CEASE_DATE.date()}  (derived, not hard-coded)')
    print(f'Right-censoring guard   : DATA_END = {DATA_END.date()}')
    print(f'Label window closes     : {(DATA_END + pd.Timedelta(days=prediction_horizon_days)).date()}')
    print(f'Buffer before max cease : {buffer_days} days')
    print()
    print(f'Overall churn rate (all data)         : {df["label"].mean():.2%}')
    print(f'Churn rate (DATA_END-clean rows only) : {df[df["datevalue"] <= DATA_END]["label"].mean():.2%}')

    df.drop('first_cease_date', axis=1, inplace=True)
    return df, DATA_END


def clean_grain(df):
    """Remove exact duplicate rows, then assert (customer, datevalue) is unique."""
    n_full_dupes = df.duplicated().sum()
    print(f'Exact duplicate rows: {n_full_dupes:,}  -> dropping')
    df = df.drop_duplicates(keep='first')

    grain = ['unique_customer_identifier', 'datevalue']
    n_grain_dupes = df.duplicated(subset=grain).sum()
    print(f'Conflicting (customer, datevalue) rows after exact dedupe: {n_grain_dupes:,}')
    if n_grain_dupes:
        df = df.sort_values(grain).drop_duplicates(subset=grain, keep='last')

    assert not df.duplicated(subset=grain).any(), 'Grain is NOT unique on (customer, datevalue)'
    print(f'Grain check passed -- {len(df):,} unique (customer, month) rows.')
    return df
