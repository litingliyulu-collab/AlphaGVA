import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from alphagen.data.expression import Feature, Ref
from alphagen.utils.random import reseed_everything
from alphagen_qlib.stock_data import FeatureType, StockData, initialize_qlib

FEATURES = [
    FeatureType.OPEN,
    FeatureType.CLOSE,
    FeatureType.HIGH,
    FeatureType.LOW,
    FeatureType.VOLUME,
    FeatureType.VWAP,
]


def build_target():
    close = Feature(FeatureType.CLOSE)
    return Ref(close, -20) / close - 1


def parse_seeds(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.replace('[', '').replace(']', '').split(',') if x.strip()]


def make_run_dir(output_root: str, model_name: str, seed: int) -> str:
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    run_dir = os.path.join(output_root, f'{model_name}_s{seed}', 'results', f'{timestamp}_ml')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def extract_feature_tensor(data: StockData) -> np.ndarray:
    start = data.max_backtrack_days
    stop = start + data.n_days
    arr = data.data[start:stop, [int(f) for f in FEATURES], :].detach().cpu().numpy()
    return np.transpose(arr, (0, 2, 1)).astype(np.float32)  # days, stocks, features


def build_arrays(data: StockData, num_lags: int) -> Tuple[np.ndarray, np.ndarray]:
    values = extract_feature_tensor(data)
    days, stocks, n_features = values.shape
    prev = np.roll(values, 1, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ret_features = values / prev - 1.0
    ret_features[0, :, :] = np.nan
    x = np.empty((days, stocks, n_features * num_lags), dtype=np.float32)
    x[:] = np.nan
    for lag in range(num_lags):
        col0 = lag * n_features
        col1 = col0 + n_features
        if lag == 0:
            x[:, :, col0:col1] = ret_features
        else:
            x[lag:, :, col0:col1] = ret_features[:-lag]
    target = build_target().evaluate(data).detach().cpu().numpy().astype(np.float32)
    return x, target


def normalize_features(x_train: np.ndarray, x_valid: np.ndarray, x_test: np.ndarray):
    flat = x_train.reshape(-1, x_train.shape[-1])
    mean = np.nanmean(flat, axis=0).astype(np.float32)
    std = np.nanstd(flat, axis=0).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0

    def transform(x):
        y = (x - mean) / std
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(y, -4.0, 4.0).astype(np.float32)

    return transform(x_train), transform(x_valid), transform(x_test)


def zscore_by_day(y: np.ndarray) -> np.ndarray:
    mean = np.nanmean(y, axis=1, keepdims=True)
    std = np.nanstd(y, axis=1, keepdims=True)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    z = (y - mean) / std
    return np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -4.0, 4.0).astype(np.float32)


def flatten_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xx = x.reshape(-1, x.shape[-1])
    yy = y.reshape(-1)
    mask = np.isfinite(yy)
    return xx[mask], yy[mask]


def daily_corr(pred: np.ndarray, label: np.ndarray, rank: bool = False) -> float:
    vals = []
    for p, y in zip(pred, label):
        mask = np.isfinite(p) & np.isfinite(y)
        if mask.sum() < 3:
            vals.append(0.0)
            continue
        pp = p[mask]
        yy = y[mask]
        if rank:
            pp = pd.Series(pp).rank(method='average').to_numpy()
            yy = pd.Series(yy).rank(method='average').to_numpy()
        sp = pp.std()
        sy = yy.std()
        if sp < 1e-12 or sy < 1e-12:
            vals.append(0.0)
        else:
            vals.append(float(np.corrcoef(pp, yy)[0, 1]))
    return float(np.nanmean(vals))


def calc_metrics(pred_test_flat: np.ndarray, y_test_raw: np.ndarray) -> Dict[str, float]:
    pred = pred_test_flat.reshape(y_test_raw.shape)
    return {
        'test/ic_mean': daily_corr(pred, y_test_raw, rank=False),
        'test/rank_ic_mean': daily_corr(pred, y_test_raw, rank=True),
    }


def write_metrics(run_dir: str, row: Dict[str, object]) -> None:
    path = os.path.join(run_dir, 'metrics.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(row, f, indent=2)


def train_lightgbm(args, seed: int, x_train, y_train, x_valid, y_valid, x_test):
    import lightgbm as lgb
    train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    valid_set = lgb.Dataset(x_valid, label=y_valid, free_raw_data=False)
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'num_leaves': args.lgb_num_leaves,
        'max_depth': args.lgb_max_depth,
        'learning_rate': args.lgb_learning_rate,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': seed,
        'feature_fraction_seed': seed,
        'bagging_seed': seed,
        'verbose': -1,
        'num_threads': args.num_threads,
    }
    model = lgb.train(
        params,
        train_set,
        valid_sets=[valid_set],
        num_boost_round=args.lgb_rounds,
        callbacks=[lgb.early_stopping(args.early_stopping), lgb.log_evaluation(args.log_every)],
    )
    pred = model.predict(x_test, num_iteration=model.best_iteration)
    return model, pred.astype(np.float32)


class MLP(torch.nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(args, seed: int, x_train, y_train, x_valid, y_valid, x_test):
    device = torch.device('cuda:0' if args.device_str == 'auto' and torch.cuda.is_available() else args.device_str)
    model = MLP(x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.mlp_lr)
    loss_fn = torch.nn.MSELoss()
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train.astype(np.float32))
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train_t, y_train_t),
        batch_size=args.mlp_batch_size,
        shuffle=True,
        generator=gen,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    for epoch in range(args.mlp_epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print({'epoch': epoch + 1, 'loss': float(np.mean(losses))}, flush=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_test), args.mlp_batch_size * 4):
            xb = torch.from_numpy(x_test[start:start + args.mlp_batch_size * 4]).to(device)
            preds.append(model(xb).detach().cpu().numpy())
    return model, np.concatenate(preds).astype(np.float32)


def run_one(args, model_name: str, seed: int) -> None:
    reseed_everything(seed)
    initialize_qlib(args.qlib_data_path, kernels=args.qlib_kernels)
    data_device = torch.device('cpu')
    train_data = StockData(args.instruments, args.train_start, args.train_end, device=data_device)
    valid_data = StockData(args.instruments, args.valid_start, args.valid_end, device=data_device)
    test_data = StockData(args.instruments, args.test_start, args.test_end, device=data_device)

    x_train_raw, y_train_raw = build_arrays(train_data, args.num_lags)
    x_valid_raw, y_valid_raw = build_arrays(valid_data, args.num_lags)
    x_test_raw, y_test_raw = build_arrays(test_data, args.num_lags)
    x_train, x_valid, x_test = normalize_features(x_train_raw, x_valid_raw, x_test_raw)

    y_train_model = zscore_by_day(y_train_raw)
    y_valid_model = zscore_by_day(y_valid_raw)
    x_train_flat, y_train_flat = flatten_xy(x_train, y_train_model)
    x_valid_flat, y_valid_flat = flatten_xy(x_valid, y_valid_model)
    x_test_flat = x_test.reshape(-1, x_test.shape[-1])

    print({'model': model_name, 'seed': seed, 'train_rows': len(x_train_flat), 'valid_rows': len(x_valid_flat), 'test_rows': len(x_test_flat), 'features': x_train_flat.shape[1]}, flush=True)
    if model_name == 'lightgbm':
        model, pred_test = train_lightgbm(args, seed, x_train_flat, y_train_flat, x_valid_flat, y_valid_flat, x_test_flat)
    elif model_name == 'mlp':
        model, pred_test = train_mlp(args, seed, x_train_flat, y_train_flat, x_valid_flat, y_valid_flat, x_test_flat)
    else:
        raise ValueError(model_name)

    run_dir = make_run_dir(args.output_root, model_name, seed)
    metrics = calc_metrics(pred_test, y_test_raw)
    row = {
        'method': model_name,
        'seed': seed,
        'num_lags': args.num_lags,
        'train_rows': len(x_train_flat),
        'valid_rows': len(x_valid_flat),
        'test_rows': len(x_test_flat),
        'features': x_train_flat.shape[1],
        'test/ic_mean': metrics['test/ic_mean'],
        'test/rank_ic_mean': metrics['test/rank_ic_mean'],
    }
    write_metrics(run_dir, row)
    np.savez_compressed(os.path.join(run_dir, 'pred_test.npz'), pred=pred_test.reshape(y_test_raw.shape), label=y_test_raw)
    if model_name == 'lightgbm':
        model.save_model(os.path.join(run_dir, 'model.txt'))
    else:
        torch.save(model.state_dict(), os.path.join(run_dir, 'model.pt'))
    print(row, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='lightgbm,mlp')
    parser.add_argument('--random_seeds', default='0')
    parser.add_argument('--output_root', default='/root/alpha_1203/gva_factor_experiments/runs_newdata/ml_baselines_manual')
    parser.add_argument('--qlib_data_path', default='/root/autodl-tmp/cn_data_akshare_2010_2026')
    parser.add_argument('--qlib_kernels', type=int, default=32)
    parser.add_argument('--device_str', default='auto')
    parser.add_argument('--instruments', default='csi300')
    parser.add_argument('--num_lags', type=int, default=60)
    parser.add_argument('--num_threads', type=int, default=8)
    parser.add_argument('--lgb_rounds', type=int, default=1000)
    parser.add_argument('--early_stopping', type=int, default=100)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lgb_num_leaves', type=int, default=210)
    parser.add_argument('--lgb_max_depth', type=int, default=8)
    parser.add_argument('--lgb_learning_rate', type=float, default=0.05)
    parser.add_argument('--mlp_epochs', type=int, default=10)
    parser.add_argument('--mlp_batch_size', type=int, default=512)
    parser.add_argument('--mlp_lr', type=float, default=1e-3)
    parser.add_argument('--train_start', default='2010-01-04')
    parser.add_argument('--train_end', default='2021-12-31')
    parser.add_argument('--valid_start', default='2022-01-04')
    parser.add_argument('--valid_end', default='2023-12-29')
    parser.add_argument('--test_start', default='2024-01-02')
    parser.add_argument('--test_end', default='2026-05-28')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    for seed in parse_seeds(args.random_seeds):
        for model_name in models:
            run_one(args, model_name, seed)
