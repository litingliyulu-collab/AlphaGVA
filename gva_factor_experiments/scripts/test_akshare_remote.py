import time
import akshare as ak

print('__AK_TEST__')
t = time.time()
spot = ak.stock_zh_a_spot_em()
print('spot_shape', spot.shape, 'sec', round(time.time() - t, 2))
print(spot.head(3).to_string(index=False))

t = time.time()
hist = ak.stock_zh_a_hist(
    symbol='600000',
    period='daily',
    start_date='20250101',
    end_date='20250110',
    adjust='qfq',
)
print('hist_shape', hist.shape, 'sec', round(time.time() - t, 2))
print(hist.tail(3).to_string(index=False))
