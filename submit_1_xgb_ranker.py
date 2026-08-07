"""
CANDIDATE #1 (best-ranked): XGBoost listwise ranking objective on the full,
uncompressed macro feature set.

Walk-forward CV score: +0.213 (mean Spearman rank correlation, 47 folds)
Diversity check: 6 distinct predicted rank permutations across 47 folds,
no single permutation over 47% -- genuinely conditional on the macro state,
not an echo of the historical base rate (see model_results.md for the
degeneracy check methodology and why it matters at this sample size).

Why this is ranked #1 among all candidates tried: it is the only top scorer
that carries NEITHER of the two structural risks found elsewhere in this
project --
  (a) no PCA compression, so it can't discard the low-variance components
      that turned out to carry the 2023 regime-shift signal (the failure
      mode that hurt `regress_rf__pca8_depth3` on the real public
      leaderboard: CV 0.192 -> real 0.083); and
  (b) no reliance on a single macro LEVEL feature sitting in unprecedented
      territory, so it isn't exposed to the confident-extrapolation failure
      that produced the HMM's real -0.0125 score.

Approach: reshape each month into 3 rows (one per factor: Momentum,
Quality, Value) sharing the same macro features plus a factor-identity
flag, and let XGBoost's `rank:pairwise` objective learn to order the 3
rows within each month directly -- rather than predicting each factor's
return independently and sorting afterwards (regress-then-rank), or
classifying a discrete rank label (classify-then-repair).
"""

import numpy as np
import pandas as pd
import xgboost as xgb

RANK_COLS = ['rank_momentum', 'rank_quality', 'rank_value']
FWD_COLS = ['fwd_momentum', 'fwd_quality', 'fwd_value']
FACTORS = ['momentum', 'quality', 'value']

KEY_FEATURES = [
    'fed_assets', 'us_bond_yield', 'us_credit_spread', 'market_volatility',
    'market_momentum', 'us_payrolls', 'us_yield_curve', 'us_cpi',
    'stock_price_strength', 'bitcoin', 'gold', 'silver'
]


def build_causal_features(raw_df):
    """
    Adds momentum / rolling-volatility / rolling-mean / interaction /
    expanding-regime-dummy columns. Every added column at row t is a
    function of rows <= t only (diff/rolling/expanding), so it is safe to
    compute in one vectorized pass over the full date-sorted train+test
    timeline -- a rolling mean at row t never "sees" row t+1, regardless of
    whether it's computed all at once or incrementally.
    """
    df = raw_df.sort_values('date').reset_index(drop=True).copy()

    for f in KEY_FEATURES:
        df[f'{f}_momentum_3m'] = df[f].diff(3)
        df[f'{f}_volatility_6m'] = df[f].rolling(6, min_periods=2).std()
        df[f'{f}_ma_3m'] = df[f].rolling(3, min_periods=1).mean()

    df['fed_loose_x_market_mom'] = df['fed_assets'] * df['market_momentum']
    df['credit_stress_x_volatility'] = df['us_credit_spread'] * df['market_volatility']
    df['valuation_x_sentiment'] = df['earnings_yield_gap'] * df['market_momentum']
    commodities = [c for c in ['bitcoin', 'gold', 'silver', 'crude', 'copper'] if c in df.columns]
    df['commodity_strength'] = df[commodities].mean(axis=1)
    df['fed_tightness'] = 1 - df['fed_assets']
    df['risk_appetite'] = 1 - df['us_credit_spread']

    # Expanding (causal) regime thresholds -- NOT a fixed full-sample
    # constant, which would let later rows leak into earlier ones.
    df['fed_tightening'] = (df['fed_assets'] < df['fed_assets'].expanding().median()).astype(int)
    df['credit_stress'] = (df['us_credit_spread'] > df['us_credit_spread'].expanding().median()).astype(int)
    df['high_volatility'] = (df['market_volatility'] > df['market_volatility'].expanding().median()).astype(int)
    df['high_yields'] = (df['us_bond_yield'] > df['us_bond_yield'].expanding().median()).astype(int)
    if 'stock_breadth_vol' in df.columns:
        df['strong_breadth'] = (df['stock_breadth_vol'] > df['stock_breadth_vol'].expanding().median()).astype(int)

    return df


def get_full_safe_feature_set(df_columns):
    engineered = ([f'{f}_momentum_3m' for f in KEY_FEATURES]
                  + [f'{f}_volatility_6m' for f in KEY_FEATURES]
                  + [f'{f}_ma_3m' for f in KEY_FEATURES])
    interactions = ['fed_loose_x_market_mom', 'credit_stress_x_volatility', 'valuation_x_sentiment',
                     'commodity_strength', 'fed_tightness', 'risk_appetite']
    regime = ['fed_tightening', 'credit_stress', 'high_volatility', 'high_yields', 'strong_breadth']
    raw40 = [c for c in df_columns if c not in RANK_COLS + FWD_COLS + ['date']
             and c not in engineered + interactions + regime]
    cols = raw40 + engineered + interactions + regime
    return [c for c in cols if c in df_columns]


def to_long_format(df, feature_cols):
    """One row per (month, factor) with a factor-identity flag, for XGBRanker."""
    rows = []
    for _, row in df.iterrows():
        for fi, f in enumerate(FACTORS):
            r = {c: row[c] for c in feature_cols}
            r['factor_id'] = fi
            if f'rank_{f}' in df.columns and not pd.isna(row.get(f'rank_{f}', np.nan)):
                r['relevance'] = 3 - int(row[f'rank_{f}'])  # rank1->2, rank2->1, rank3->0
            rows.append(r)
    return pd.DataFrame(rows)


def scores_to_ranks(scores):
    order = sorted(scores.keys(), key=lambda f: -scores[f])
    return {f: i + 1 for i, f in enumerate(order)}


def main():
    train_raw = pd.read_csv('train.csv', parse_dates=['date'])
    test_raw = pd.read_csv('test.csv', parse_dates=['date'])

    # Concatenate train+test in date order so test's rolling/momentum
    # features get real lookback from train's tail, instead of NaN warm-up.
    # test.csv has no rank_*/fwd_* at all -- these stay NaN for test rows
    # and are never used as inputs, only as the (absent) labels we predict.
    train_marked, test_marked = train_raw.copy(), test_raw.copy()
    for col in RANK_COLS + FWD_COLS:
        if col not in test_marked.columns:
            test_marked[col] = np.nan
    combined = pd.concat([train_marked, test_marked], ignore_index=True).sort_values('date').reset_index(drop=True)
    combined = build_causal_features(combined)

    final_train = combined[combined['date'].isin(train_raw['date'])].reset_index(drop=True)
    final_test = combined[combined['date'].isin(test_raw['date'])].reset_index(drop=True)

    feature_cols = get_full_safe_feature_set(combined.columns)
    # Drop the leading warm-up rows (insufficient rolling-window history) --
    # there's no valid data to fill them with, so we drop rather than
    # forward/back-fill, which would risk leaking later rows into earlier ones.
    final_train = final_train.dropna(subset=feature_cols).reset_index(drop=True)
    assert final_test[feature_cols].isnull().sum().sum() == 0, \
        "test features should have full lookback from train's tail -- unexpected NaN"

    ltr_cols = feature_cols + ['factor_id']
    long_train = to_long_format(final_train, feature_cols)
    groups = [3] * len(final_train)  # each month is one "query" of 3 "documents" (factors)

    ranker = xgb.XGBRanker(objective='rank:pairwise', max_depth=3, n_estimators=80,
                             learning_rate=0.05, min_child_weight=4, random_state=0, verbosity=0)
    ranker.fit(long_train[ltr_cols].values, long_train['relevance'].values, group=groups)

    final_ranks = []
    for _, row in final_test.iterrows():
        long_target = to_long_format(row.to_frame().T, feature_cols)
        scores = ranker.predict(long_target[ltr_cols].values)
        final_ranks.append(scores_to_ranks({FACTORS[i]: float(scores[i]) for i in range(3)}))

    submission = pd.DataFrame({
        'date': final_test['date'].dt.strftime('%Y-%m-%d'),
        'rank_momentum': [r['momentum'] for r in final_ranks],
        'rank_quality': [r['quality'] for r in final_ranks],
        'rank_value': [r['value'] for r in final_ranks],
    })
    assert submission[['rank_momentum', 'rank_quality', 'rank_value']].apply(
        lambda r: sorted(r.values.tolist()) == [1, 2, 3], axis=1).all(), "invalid permutation produced"

    submission.to_csv('submission_1_xgb_ranker.csv', index=False)
    print("Saved submission_1_xgb_ranker.csv")
    print(submission['rank_momentum'].value_counts().sort_index().rename('rank_momentum').to_string())
    print(submission['rank_quality'].value_counts().sort_index().rename('rank_quality').to_string())
    print(submission['rank_value'].value_counts().sort_index().rename('rank_value').to_string())


if __name__ == '__main__':
    main()
