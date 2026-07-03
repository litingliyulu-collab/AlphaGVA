from alphagen_qlib.stock_data import StockData

sd = StockData(
    instrument="csi500",
    start_time="2024-01-02",
    end_time="2024-01-10",
    max_backtrack_days=60,
    max_future_days=20,
)
print("features", sd.n_features, "stocks", sd.n_stocks, "days", sd.n_days)
print("dates", sd._dates[0], sd._dates[-1])
