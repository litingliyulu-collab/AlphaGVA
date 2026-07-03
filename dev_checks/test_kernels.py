import qlib
from qlib.config import REG_CN, C
qlib.init(provider_uri='/root/autodl-tmp/cn_data_akshare_2010_2026', region=REG_CN, kernels=1)
print('kernels', getattr(C, 'kernels', None), C.get('kernels', None) if hasattr(C, 'get') else None)
