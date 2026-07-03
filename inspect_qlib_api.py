import inspect
import qlib
from qlib.config import REG_CN
qlib.init(provider_uri='/root/autodl-tmp/cn_data', region=REG_CN)
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data import D
print('QlibDataLoader.load', inspect.signature(QlibDataLoader.load))
print('D.features', inspect.signature(D.features))
print('D.dataset', inspect.signature(D.dataset))
print('D.calendar', inspect.signature(D.calendar))
