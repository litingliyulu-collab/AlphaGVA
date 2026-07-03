import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path('/root/autodl-tmp/gva_factor_experiments/backtests/filter_suite_with_locked_top50_20260629')
aff = Path('/root/autodl-tmp/aff_official_workspace_20260629/out/affofficial_csi300_2023_0/aff_official_daily_report.csv')
out = base
methods = {
    'AlphaGen-PPO': ['alphagen_ppo_s0','alphagen_ppo_s1','alphagen_ppo_s2'],
    'Critic-GVA': ['critic_gva_s0','critic_gva_s1','critic_gva_s2'],
    'Full-GVA': ['full_gva_s0','full_gva_s1','full_gva_s2'],
    'Locked-WarmPool-GVA': ['locked_warmpool_gva_s0','locked_warmpool_gva_s1','locked_warmpool_gva_s2'],
    'PPO_filter_strong': ['ppo_filter_s0','ppo_filter_s1','ppo_filter_s2'],
    'GP_filter_weak': ['gp_filter_weak_s0','gp_filter_weak_s1','gp_filter_weak_s2'],
}

def read_method(keys, col):
    xs=[]
    for k in keys:
        p = base / k / 'daily_report.csv'
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df = df.loc['2025-01-02':'2025-12-31']
        xs.append(df[col].rename(k))
    if not xs:
        return None
    return pd.concat(xs, axis=1).fillna(0).mean(axis=1)

aff_df = pd.read_csv(aff, parse_dates=['date']).set_index('date')
for col, aff_col, fname, title, ylabel in [
    ('return', 'return', 'cumulative_return_2025_with_aff.png', 'Cumulative Return on 2025 Test Window', 'Cumulative Return'),
    ('excess_return', 'excess_return', 'cumulative_excess_return_2025_with_aff.png', 'Cumulative Excess Return on 2025 Test Window', 'Cumulative Excess Return'),
]:
    plt.figure(figsize=(12,6))
    for name, keys in methods.items():
        r = read_method(keys, col)
        if r is None or r.empty:
            continue
        curve = (1+r).cumprod()-1
        lw = 3 if name in ['Full-GVA','Locked-WarmPool-GVA'] else 1.8
        plt.plot(curve.index, curve.values, label=name, linewidth=lw)
    ar = aff_df[aff_col].loc['2025-01-02':'2025-12-31'].fillna(0)
    plt.plot(ar.index, (1+ar).cumprod()-1, label='AFF-official-adapted', linewidth=2.5, linestyle='--')
    if col == 'return':
        bench = aff_df['benchmark_return'].loc['2025-01-02':'2025-12-31'].fillna(0)
        plt.plot(bench.index, (1+bench).cumprod()-1, label='CSI300 universe EW', color='black', linestyle=':', linewidth=2)
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / fname, dpi=180)
    plt.close()
print(out / 'cumulative_return_2025_with_aff.png')
print(out / 'cumulative_excess_return_2025_with_aff.png')
