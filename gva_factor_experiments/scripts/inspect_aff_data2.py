from gan.utils.data import get_data_by_year
ret = get_data_by_year(train_start=2011, train_end=2023, valid_year=2024, test_year=2025, instruments='csi300', target='normal', freq='day')
data_test = ret[4]
for name in ['data','features','_features','_dates','_stock_ids','stock_ids','instrument']:
    if hasattr(data_test,name):
        v=getattr(data_test,name)
        print(name, type(v), getattr(v,'shape',None), v if name in ['features','_features'] else '')
