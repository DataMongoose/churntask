import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve


def plot_performance_over_time(results_df, save_path='rolling_window_performance.png'):
    """Three-panel chart: ROC-AUC over time, PR-AUC over time, churn rate + volume."""
    months = results_df['test_month']
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(months, results_df['boost_auc'], marker='o', lw=2, color='C0',
                 label=f'Boosting (mean={results_df["boost_auc"].mean():.3f})')
    axes[0].plot(months, results_df['xgb_auc'],   marker='^', lw=2, color='C2',
                 label=f'XGBoost  (mean={results_df["xgb_auc"].mean():.3f})')
    axes[0].plot(months, results_df['lr_auc'],    marker='s', lw=2, color='C1',
                 label=f'LR baseline (mean={results_df["lr_auc"].mean():.3f})')
    for col, c in [('boost_auc','C0'), ('xgb_auc','C2'), ('lr_auc','C1')]:
        axes[0].axhline(results_df[col].mean(), color=c, linestyle='--', alpha=0.5)
    axes[0].set_ylabel('ROC-AUC'); axes[0].set_ylim(0.5, 1.0)
    axes[0].legend(); axes[0].set_title('Model Performance Over Time — Walk-Forward CV (3 models)')

    axes[1].plot(months, results_df['boost_pr'], marker='o', lw=2, color='C0',
                 label=f'Boosting (mean={results_df["boost_pr"].mean():.3f})')
    axes[1].plot(months, results_df['xgb_pr'],   marker='^', lw=2, color='C2',
                 label=f'XGBoost  (mean={results_df["xgb_pr"].mean():.3f})')
    axes[1].plot(months, results_df['lr_pr'],    marker='s', lw=2, color='C1',
                 label=f'LR baseline (mean={results_df["lr_pr"].mean():.3f})')
    for col, c in [('boost_pr','C0'), ('xgb_pr','C2'), ('lr_pr','C1')]:
        axes[1].axhline(results_df[col].mean(), color=c, linestyle='--', alpha=0.5)
    axes[1].set_ylabel('PR-AUC'); axes[1].legend()

    ax2b = axes[2].twinx()
    axes[2].bar(months, results_df['n_test'], width=20, color='#95a5a6', alpha=0.4, label='n test')
    ax2b.plot(months, results_df['churn_rate'] * 100, marker='^', color='#e67e22',
              lw=2, label='churn rate %')
    axes[2].set_ylabel('Customers in test month'); ax2b.set_ylabel('Churn rate (%)')
    axes[2].set_xlabel('Test month')
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_roc_pr_score_dist(all_y, boost_all, xgb_all, lr_all,
                           save_path='model_evaluation.png'):
    """ROC curve, PR curve, and score distribution — pooled OOS."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for scores, label, color in [(boost_all, 'Boosting', 'C0'),
                                  (xgb_all,   'XGBoost',  'C2'),
                                  (lr_all,    'LR',       'C1')]:
        fpr, tpr, _ = roc_curve(all_y, scores)
        auc = roc_auc_score(all_y, scores)
        axes[0].plot(fpr, tpr, lw=2, color=color, label=f'{label} AUC={auc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].set_title('ROC Curve (pooled OOS)'); axes[0].legend()

    baseline = all_y.mean()
    for scores, label, color in [(boost_all, 'Boosting', 'C0'),
                                  (xgb_all,   'XGBoost',  'C2'),
                                  (lr_all,    'LR',       'C1')]:
        prec, rec, _ = precision_recall_curve(all_y, scores)
        pr = average_precision_score(all_y, scores)
        axes[1].plot(rec, prec, lw=2, color=color, label=f'{label} PR={pr:.3f}')
    axes[1].axhline(baseline, color='grey', linestyle='--', label=f'Baseline={baseline:.3f}')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall (pooled OOS)'); axes[1].legend()

    for scores, label, lcolor in [(boost_all, 'Boosting', 'C0'), (xgb_all, 'XGBoost', 'C2')]:
        axes[2].hist(scores[all_y == 0], bins=60, alpha=0.45, density=True,
                     color=lcolor, label=f'{label} Stay')
        axes[2].hist(scores[all_y == 1], bins=60, alpha=0.45, density=True,
                     color=lcolor, histtype='step', linewidth=2,
                     label=f'{label} Churn', linestyle='--')
    axes[2].set_xlabel('Predicted churn probability'); axes[2].set_ylabel('Density')
    axes[2].set_title('Score Distribution (pooled OOS)'); axes[2].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_confusion_matrices(all_y, boost_all, thresholds=(0.30, 0.50),
                            save_path='confusion_matrices.png'):
    """Confusion matrices at two operating thresholds."""
    fig, axes = plt.subplots(1, len(thresholds), figsize=(6 * len(thresholds), 5))
    for ax, thresh in zip(axes, thresholds):
        pred = (boost_all >= thresh).astype(int)
        cm   = confusion_matrix(all_y, pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Stay', 'Churn'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='')
        ax.set_title(f'Boosting — Threshold = {thresh:.2f}', fontsize=12)
        total = cm.sum()
        for text_obj, val in zip(disp.text_.ravel(), cm.ravel()):
            pct = val / total * 100
            text_obj.set_text(f'{val:,.0f}\n({pct:.1f}%)')
            text_obj.set_fontsize(10)
    plt.suptitle('Confusion Matrices — Boosting, Pooled OOS', fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_lift_curves(all_y, boost_all, lr_all, rule_points=None, save_path='lift_curves.png'):
    """Cumulative gain and lift curves.

    rule_points: optional list of (pct_flagged, pct_captured, label) tuples.
    Each is plotted as a diamond marker so a naive rule can be compared directly.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for scores, label, color in [(boost_all, 'Boosting', 'C0'), (lr_all, 'LR', 'C1')]:
        order     = np.argsort(scores)[::-1]
        y_sorted  = all_y[order]
        n         = len(y_sorted)
        total_pos = all_y.sum()
        cum_pos   = np.cumsum(y_sorted)
        pct_pop   = np.arange(1, n + 1) / n
        cum_gain  = cum_pos / total_pos
        lift      = cum_gain / pct_pop
        axes[0].plot(pct_pop * 100, cum_gain * 100, lw=2, color=color, label=label)
        axes[1].plot(pct_pop * 100, lift,            lw=2, color=color, label=label)

    axes[0].plot([0, 100], [0, 100], 'k--', lw=1, label='Random baseline')
    axes[1].axhline(1.0, color='k', linestyle='--', lw=1, label='Random (lift = 1×)')
    for pct in [10, 20, 30]:
        axes[0].axvline(pct, color='grey', alpha=0.3, lw=0.8, linestyle=':')
    for d in range(10, 110, 10):
        axes[1].axvline(d, color='grey', alpha=0.2, lw=0.5)

    if rule_points:
        rule_colors = ['C2', 'C3', 'C4']
        for (pct_flag, pct_caught, rlabel), rc in zip(rule_points, rule_colors):
            lift_val = pct_caught / pct_flag if pct_flag > 0 else 1.0
            axes[0].scatter([pct_flag * 100], [pct_caught * 100], s=140, zorder=5,
                            marker='D', color=rc, label=rlabel, edgecolors='k', linewidths=0.8)
            axes[1].scatter([pct_flag * 100], [lift_val], s=140, zorder=5,
                            marker='D', color=rc, label=rlabel, edgecolors='k', linewidths=0.8)

    axes[0].set_xlabel('% Population Contacted (ranked by score, highest first)')
    axes[0].set_ylabel('% Churners Captured')
    axes[0].set_title('Cumulative Gain Curve (pooled OOS)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('% Population Contacted (ranked by score, highest first)')
    axes[1].set_ylabel('Lift over Random')
    axes[1].set_title('Cumulative Lift Curve (pooled OOS)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_calibration(all_y, boost_all, iso_cal, save_path='calibration_curve.png'):
    """Reliability diagram: raw balanced-weight scores vs isotonic-calibrated."""
    cal_scores = iso_cal.predict(boost_all)
    fig, ax = plt.subplots(figsize=(6, 6))
    for s, lab, c in [(boost_all,  'raw (balanced) scores', 'C3'),
                      (cal_scores, 'isotonic-calibrated',   'C0')]:
        frac_pos, mean_pred = calibration_curve(all_y, s, n_bins=10, strategy='quantile')
        ax.plot(mean_pred, frac_pos, marker='o', lw=2, color=c, label=lab)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='perfectly calibrated')
    ax.set_xlabel('Mean predicted value'); ax.set_ylabel('Observed churn fraction')
    ax.set_title('Reliability Diagram (pooled OOS holdout)'); ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_score_distributions(all_y, boost_all, xgb_all, lr_all,
                             save_path='score_distributions_imbalance.png'):
    """Score distributions by model showing separation and default threshold placement."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
    for ax, scores, title, color in [
        (axes[0], boost_all, 'HistGBM\n(class_weight=balanced)', 'C0'),
        (axes[1], xgb_all,   'XGBoost\n(scale_pos_weight=7)',    'C2'),
        (axes[2], lr_all,    'LR\n(class_weight=balanced)',       'C1'),
    ]:
        ax.hist(scores[all_y == 0], bins=60, alpha=0.55, density=True, color=color, label='Stay')
        ax.hist(scores[all_y == 1], bins=60, alpha=0.55, density=True, color='C3',  label='Churn')
        ax.axvline(0.50,         color='black', linestyle='--', lw=1.2, label='t=0.50 (default)')
        ax.axvline(all_y.mean(), color='grey',  linestyle=':',  lw=1.0,
                   label=f'base rate {all_y.mean():.1%}')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Raw predicted score'); ax.set_ylabel('Density')
        ax.legend(fontsize=7)
    plt.suptitle(
        'Score distributions by model (raw balanced-weight scores)\n'
        'Both classes pushed apart from base rate — default t=0.5 sits in a void for all models.',
        fontsize=10, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_threshold_analysis(thresh_df, t_f1, t_cost, all_y,
                            cost_fp, cost_fn, save_path='threshold_optimisation.png'):
    """Three-panel threshold analysis: F1 score, precision-recall, business cost."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(thresh_df['threshold'], thresh_df['f1'], lw=2, color='C0', label='F1 (β=1)')
    axes[0].axvline(t_f1, color='C0', linestyle='--', alpha=0.6, lw=1, label=f'Max F1 t={t_f1:.2f}')
    axes[0].axvline(0.5, color='grey', linestyle=':', lw=1, label='Default t=0.50')
    axes[0].set_xlabel('Threshold'); axes[0].set_ylabel('F1 score')
    axes[0].set_title('F1 vs Threshold\n(Boosting, pooled OOS)')
    axes[0].legend(fontsize=8); axes[0].set_xlim(0.02, 0.80); axes[0].set_ylim(0, 0.75)

    axes[1].plot(thresh_df['threshold'], thresh_df['precision'], lw=2, color='C3', label='Precision')
    axes[1].plot(thresh_df['threshold'], thresh_df['recall'],    lw=2, color='C1', label='Recall')
    axes[1].fill_between(thresh_df['threshold'], thresh_df['precision'],
                         thresh_df['recall'], alpha=0.08, color='purple')
    axes[1].axvline(t_cost, color='black', linestyle='--', lw=1.5,
                    label=f'Min-cost t={t_cost:.2f}')
    axes[1].axvline(0.5, color='grey', linestyle=':', lw=1, label='Default t=0.50')
    axes[1].axhline(all_y.mean(), color='grey', linestyle=':', lw=0.8,
                    label=f'Base rate {all_y.mean():.1%}')
    axes[1].set_xlabel('Threshold'); axes[1].set_ylabel('Score')
    axes[1].set_title('Precision & Recall vs Threshold\n(Boosting, pooled OOS)')
    axes[1].legend(fontsize=8); axes[1].set_xlim(0.02, 0.80)

    cost_min_val = thresh_df['business_cost'].min()
    cost_at_05   = thresh_df.iloc[(thresh_df['threshold'] - 0.5).abs().argsort().iloc[0]]['business_cost']
    axes[2].plot(thresh_df['threshold'], thresh_df['business_cost'] / 1e6, lw=2, color='C5')
    axes[2].axvline(t_cost, color='black', linestyle='--', lw=1.5,
                    label=f'Optimal t={t_cost:.2f}  (£{cost_min_val/1e6:.2f}M)')
    axes[2].axvline(0.5, color='grey', linestyle=':', lw=1,
                    label=f'Default t=0.50  (£{cost_at_05/1e6:.2f}M)')
    axes[2].axhline(cost_min_val / 1e6, color='black', linestyle=':', lw=0.8, alpha=0.5)
    axes[2].set_xlabel('Threshold'); axes[2].set_ylabel('Business cost (£M)')
    axes[2].set_title(f'Business Cost vs Threshold\nFP=£{cost_fp} | FN=£{cost_fn} | ratio 1:{cost_fn//cost_fp}')
    axes[2].legend(fontsize=8); axes[2].set_xlim(0.02, 0.80)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_feature_importance(fi_df, test_month_label, top_n=20,
                            save_path='feature_importance_boost.png'):
    """Horizontal bar chart of permutation feature importance (top N)."""
    top = fi_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top['feature'][::-1], top['importance_mean'][::-1],
            xerr=top['importance_std'][::-1], color='#3498db', ecolor='grey')
    ax.set_xlabel('Permutation importance (mean ROC-AUC drop, held-out month)')
    ax.set_title(f'Top {top_n} Features - Boosting (permuted on held-out {test_month_label})')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_shap_summary(shap_values, X, top_n=20, save_path='shap_summary.png'):
    """SHAP beeswarm — direction + magnitude for the boosting model."""
    import shap
    shap.summary_plot(shap_values, X, max_display=top_n, show=False)
    plt.title(f'Top {top_n} Features — SHAP Beeswarm (Boosting Model, held-out month)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_risk_band_profiles(df_score, profile_cols=None, save_path='risk_band_profiles.png'):
    """Median feature value by risk tier — shows what drives each risk band."""
    if profile_cols is None:
        profile_cols = [
            'tenure_days', 'ooc_days', 'contract_dd_cancels', 'dd_cancel_60_day',
            'calls_3m_count', 'calls_3m_loyalty', 'usage_3m_avg_download', 'speed_gap',
        ]
    tier_order = ['P4 Monitor', 'P3 Email/SMS', 'P2 Outbound', 'P1 Call first']
    palette    = {'P4 Monitor': '#2ecc71', 'P3 Email/SMS': '#f39c12',
                  'P2 Outbound': '#e67e22', 'P1 Call first': '#e74c3c'}

    fig, axes = plt.subplots(2, 4, figsize=(16, 7)); axes = axes.flatten()
    for ax, col in zip(axes, profile_cols):
        data = df_score[[col, 'risk_tier']].dropna()
        grp  = data.groupby('risk_tier', observed=True)[col].median().reindex(tier_order)
        ax.bar(grp.index, grp.values, color=[palette[t] for t in grp.index])
        ax.set_title(col, fontsize=9); ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=30)
    plt.suptitle('Median feature values by risk tier', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
