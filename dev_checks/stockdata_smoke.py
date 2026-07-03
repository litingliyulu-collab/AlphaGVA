import torch
from alphagen_qlib.stock_data import initialize_qlib, StockData
initialize_qlib('/root/autodl-tmp/cn_data_akshare_2010_2026')
for inst in ['csi300', 'csi500', 'csi1000']:
    d = StockData(inst, '2024-01-02', '2024-02-01', device=torch.device('cpu'))
    print(inst, 'data', tuple(d.data.shape), 'n_days', d.n_days, 'n_stocks', d.n_stocks, 'first', d._dates[0], 'last', d._dates[-1])
