import akshare as ak
for symbol in ['000300', '000905', '000852']:
    print('__INDEX__', symbol)
    try:
        df = ak.index_stock_cons_csindex(symbol=symbol)
        print('shape', df.shape)
        print(df.head(3).to_string(index=False))
    except Exception as e:
        print('ERR', type(e).__name__, e)
