import qlib
from qlib.config import REG_CN, C
qlib.init(provider_uri='/root/autodl-tmp/cn_data_akshare_2010_2026', region=REG_CN)
print(getattr(C, 'kernels', None))
