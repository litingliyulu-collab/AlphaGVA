import qlib
from qlib.config import REG_CN
qlib.init(provider_uri='/root/autodl-tmp/cn_data', region=REG_CN)
from qlib.data import D
inst = D.instruments('csi300')
print('inst type', type(inst), inst if isinstance(inst, str) else 'not str')
try:
    df = D.features(inst, ['$close'], '2002-01-01', '2002-01-10', freq='day', inst_processors=[])
    print('D.features ok', df.shape, df.head())
except Exception as e:
    import traceback; traceback.print_exc()
try:
    df = D.features('csi300', ['$close'], '2002-01-01', '2002-01-10', freq='day')
    print('D.features string ok', df.shape, df.head())
except Exception as e:
    import traceback; traceback.print_exc()
