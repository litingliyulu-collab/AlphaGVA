"""
聚合多个随机种子下的实验结果，计算均值和置信区间
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Union
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import seaborn as sns

def find_result_dirs(base_dir: str, pattern: str = None) -> List[str]:
    """
    查找所有实验结果目录
    
    Args:
        base_dir: 结果根目录（如 "./out_more/results"）
        pattern: 目录名模式（如 "csi300_10_*"）
    
    Returns:
        结果目录列表
    """
    base_path = Path(base_dir)
    if pattern:
        # 使用glob模式匹配
        dirs = list(base_path.glob(pattern))
    else:
        dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    return [str(d) for d in dirs]

def load_metrics_from_dir(result_dir: str) -> pd.DataFrame:
    """
    从单个结果目录加载指标CSV文件
    
    Args:
        result_dir: 结果目录路径
    
    Returns:
        包含所有指标的DataFrame
    """
    metrics_file = os.path.join(result_dir, 'metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"Warning: {metrics_file} not found, skipping...")
        return None
    
    df = pd.read_csv(metrics_file)
    # 添加seed信息（从目录名提取）
    dir_name = os.path.basename(result_dir)
    # 假设目录名格式为: instruments_pool_seed_timestamp_tag
    parts = dir_name.split('_')
    if len(parts) >= 3:
        try:
            df['seed'] = int(parts[2])
        except ValueError:
            df['seed'] = 0
    else:
        df['seed'] = 0
    df['result_dir'] = result_dir
    return df

def aggregate_results(result_dirs: List[str]) -> pd.DataFrame:
    """
    聚合多个随机种子的结果
    
    Args:
        result_dirs: 结果目录列表
    
    Returns:
        聚合后的DataFrame，包含均值和标准差
    """
    all_dfs = []
    for result_dir in result_dirs:
        df = load_metrics_from_dir(result_dir)
        if df is not None:
            all_dfs.append(df)
    
    if not all_dfs:
        raise ValueError("No valid result directories found!")
    
    # 合并所有DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # 按timestep分组，计算统计量
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['seed']]  # 排除seed列
    
    aggregated = []
    for timestep in sorted(combined_df['timestep'].unique()):
        timestep_data = combined_df[combined_df['timestep'] == timestep]
        
        row = {'timestep': timestep}
        for col in numeric_cols:
            values = timestep_data[col].dropna()
            if len(values) > 0:
                row[f'{col}_mean'] = values.mean()
                row[f'{col}_std'] = values.std()
                row[f'{col}_min'] = values.min()
                row[f'{col}_max'] = values.max()
                # 计算95%置信区间（使用t分布）
                if len(values) > 1:
                    try:
                        ci = stats.t.interval(0.95, len(values)-1, 
                                             loc=values.mean(), 
                                             scale=stats.sem(values))
                        row[f'{col}_ci_lower'] = ci[0]
                        row[f'{col}_ci_upper'] = ci[1]
                    except (ValueError, RuntimeWarning):
                        # 如果计算失败，使用均值±标准差
                        row[f'{col}_ci_lower'] = values.mean() - values.std()
                        row[f'{col}_ci_upper'] = values.mean() + values.std()
                else:
                    row[f'{col}_ci_lower'] = values.mean()
                    row[f'{col}_ci_upper'] = values.mean()
        
        aggregated.append(row)
    
    return pd.DataFrame(aggregated).sort_values('timestep')

def plot_with_confidence_interval(
    df: pd.DataFrame,
    metric: str,
    save_path: str = None,
    title: str = None,
    ylabel: str = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    绘制带置信区间的指标曲线
    
    Args:
        df: 聚合后的DataFrame
        metric: 指标名称（如 'test/ic_mean'）
        save_path: 保存路径
        title: 图表标题
        ylabel: Y轴标签
        figsize: 图表大小
    """
    mean_col = f'{metric}_mean'
    ci_lower_col = f'{metric}_ci_lower'
    ci_upper_col = f'{metric}_ci_upper'
    
    if mean_col not in df.columns:
        print(f"Warning: {mean_col} not found in DataFrame")
        return
    
    plt.figure(figsize=figsize)
    
    x = df['timestep']
    y_mean = df[mean_col]
    y_lower = df[ci_lower_col]
    y_upper = df[ci_upper_col]
    
    # 绘制置信区间（填充区域）
    plt.fill_between(x, y_lower, y_upper, alpha=0.3, label='95% Confidence Interval')
    
    # 绘制均值曲线
    plt.plot(x, y_mean, linewidth=2, label='Mean')
    
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel(ylabel or metric, fontsize=12)
    plt.title(title or f'{metric} with 95% Confidence Interval', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

def main(
    result_base_dir: str = "./out_more/results",
    pattern: str = None,
    output_dir: str = "./out_more/aggregated_results",
    metrics_to_plot: str = None  # 改为字符串，然后解析
):
    """
    主函数：聚合结果并生成可视化
    
    Args:
        result_base_dir: 结果根目录
        pattern: 目录名模式（如 "csi300_10_*_rl"）
        output_dir: 输出目录
        metrics_to_plot: 要绘制的指标列表（逗号分隔的字符串，如 "test/ic_mean,test/rank_ic_mean"）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== 修复：解析 metrics_to_plot 字符串 ==========
    if metrics_to_plot is None:
        metrics_list = [
            'test/ic_mean',
            'test/rank_ic_mean',
            'pool/best_ic_ret',
            'rollout/ep_rew_mean',
            'train/value_loss',
        ]
    else:
        # 将逗号分隔的字符串转换为列表
        metrics_list = [m.strip() for m in metrics_to_plot.split(',') if m.strip()]
    
    print(f"Metrics to plot: {metrics_list}")
    
    # 查找所有结果目录
    result_dirs = find_result_dirs(result_base_dir, pattern)
    print(f"Found {len(result_dirs)} result directories")
    
    if len(result_dirs) == 0:
        print("No result directories found!")
        return
    
    # 聚合结果
    print("Aggregating results...")
    aggregated_df = aggregate_results(result_dirs)
    
    # 保存聚合结果
    output_file = os.path.join(output_dir, 'aggregated_metrics.csv')
    aggregated_df.to_csv(output_file, index=False)
    print(f"Saved aggregated results to {output_file}")
    print(f"DataFrame shape: {aggregated_df.shape}")
    print(f"Available columns: {list(aggregated_df.columns)[:10]}...")  # 打印前10列
    
    # 绘制每个指标
    print("Generating plots...")
    plots_generated = 0
    for metric in metrics_list:
        mean_col = f'{metric}_mean'
        if mean_col in aggregated_df.columns:
            plot_path = os.path.join(output_dir, f'{metric.replace("/", "_")}.png')
            plot_with_confidence_interval(
                aggregated_df,
                metric,
                save_path=plot_path,
                title=f'{metric} (Mean ± 95% CI)'
            )
            plots_generated += 1
        else:
            print(f"Warning: {metric} (column {mean_col}) not found in aggregated results")
            # 打印相似的列名帮助调试
            similar_cols = [c for c in aggregated_df.columns if metric.split('/')[-1] in c]
            if similar_cols:
                print(f"  Similar columns found: {similar_cols[:5]}")
    
    print(f"Generated {plots_generated} plots")
    
    # 保存统计摘要
    summary = {
        'num_seeds': len(result_dirs),
        'result_dirs': result_dirs,
        'metrics': list(aggregated_df.columns),
        'final_metrics': {}
    }
    
    # 获取最终指标值
    if len(aggregated_df) > 0:
        final_row = aggregated_df.iloc[-1]
        for metric in metrics_list:
            mean_col = f'{metric}_mean'
            if mean_col in aggregated_df.columns:
                summary['final_metrics'][metric] = {
                    'mean': float(final_row[mean_col]),
                    'std': float(final_row[f'{metric}_std']),
                    'ci_lower': float(final_row[f'{metric}_ci_lower']),
                    'ci_upper': float(final_row[f'{metric}_ci_upper']),
                }
    
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")

if __name__ == '__main__':
    import fire
    fire.Fire(main)