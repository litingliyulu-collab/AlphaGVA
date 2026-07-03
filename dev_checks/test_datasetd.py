import qlib
from qlib.config import REG_CN
qlib.init(provider_uri='/root/autodl-tmp/cn_data_akshare_2010_2026', region=REG_CN)
from qlib.data import D
from qlib.data.data import DatasetD
inst = D.instruments('csi300')
df = DatasetD.dataset(inst, ['$close'], '2024-01-02', '2024-02-01', 'day', inst_processors=[])
print(df.shape)
print(df.head())
print(df.index.names)
