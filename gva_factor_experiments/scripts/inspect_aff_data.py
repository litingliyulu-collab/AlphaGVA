from gan.utils.data import get_data_by_year
ret = get_data_by_year(train_start=2011, train_end=2023, valid_year=2024, test_year=2025, instruments='csi300', target='normal', freq='day')
data_test = ret[4]
print(type(data_test))
for name in dir(data_test):
    if name.startswith('_') and any(k in name.lower() for k in ['data','target','date','inst','features']):
        v=getattr(data_test,name)
        print(name, type(v), getattr(v,'shape',None))
print('n_days', getattr(data_test,'n_days',None), 'n_stocks', getattr(data_test,'n_stocks',None), 'max_backtrack', getattr(data_test,'max_backtrack_days',None), 'max_future', getattr(data_test,'max_future_days',None))
