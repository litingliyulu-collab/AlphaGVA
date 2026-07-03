from typing import List, Union, Optional, Tuple
from enum import IntEnum
import numpy as np
import pandas as pd
import torch


class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5


_DEFAULT_QLIB_DATA_PATH = "~/.qlib/qlib_data/cn_data"
_QLIB_INITIALIZED = False


def initialize_qlib(qlib_data_path: str = _DEFAULT_QLIB_DATA_PATH, kernels: int = 1) -> None:
    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri=qlib_data_path, region=REG_CN, kernels=kernels)
    global _QLIB_INITIALIZED
    _QLIB_INITIALIZED = True


class StockData:
    _qlib_initialized: bool = False

    def __init__(
        self,
        instrument: Union[str, List[str]],
        start_time: str,
        end_time: str,
        max_backtrack_days: int = 100,
        max_future_days: int = 30,
        features: Optional[List[FeatureType]] = None,
        device: torch.device = torch.device("cuda:0"),
        preloaded_data: Optional[Tuple[torch.Tensor, pd.Index, pd.Index]] = None
    ) -> None:
        self._init_qlib()

        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        data_tup = preloaded_data if preloaded_data is not None else self._get_data()
        self.data, self._dates, self._stock_ids = data_tup

    @classmethod
    def _init_qlib(cls) -> None:
        global _QLIB_INITIALIZED
        if not _QLIB_INITIALIZED:
            initialize_qlib()

    def _load_exprs(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
        # This evaluates expressions on qlib data and returns a dataframe indexed
        # by (datetime, instrument). Direct D.features is more stable across qlib
        # versions than QlibDataLoader for this project.
        from qlib.data import D
        if not isinstance(exprs, list):
            exprs = [exprs]
        cal: np.ndarray = D.calendar()
        start_index = cal.searchsorted(pd.Timestamp(self._start_time))  # type: ignore
        end_index = cal.searchsorted(pd.Timestamp(self._end_time))  # type: ignore
        real_start_time = cal[max(0, start_index - self.max_backtrack_days)]
        if end_index >= len(cal):
            end_index = len(cal) - 1
        elif cal[end_index] != pd.Timestamp(self._end_time):
            end_index -= 1
        real_end_idx = min(len(cal) - 1, end_index + self.max_future_days)
        real_end_time = cal[real_end_idx]
        instruments = D.instruments(self._instrument) if isinstance(self._instrument, str) else self._instrument
        from qlib.data.data import DatasetD
        df = DatasetD.dataset(instruments, exprs, real_start_time, real_end_time, 'day', inst_processors=[])
        df.columns = exprs
        if df.index.names != ['datetime', 'instrument'] and set(df.index.names) == {'datetime', 'instrument'}:
            df = df.reorder_levels(['datetime', 'instrument']).sort_index()
        return df

    # def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
    #     features = ['$' + f.name.lower() for f in self._features]
    #     df = self._load_exprs(features)
    #     df = df.stack().unstack(level=1)
    #     dates = df.index.levels[0]                                      # type: ignore
    #     stock_ids = df.columns
    #     values = df.values
    #     values = values.reshape((-1, len(features), values.shape[-1]))  # type: ignore
    #     return torch.tensor(values, dtype=torch.float, device=self.device), dates, stock_ids
    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        features = ['$' + f.name.lower() for f in self._features]
        df = self._load_exprs(features)
        
        # 从索引中获取唯一的日期和股票
        dates = df.index.get_level_values('datetime').unique()
        stocks = df.index.get_level_values('instrument').unique()
        
        # 直接reshape为3D数组
        n_days = len(dates)
        n_stocks = len(stocks)
        n_features = len(features)
        
        # 检查维度是否匹配
        if df.shape[0] == n_days * n_stocks:
            values = df.values.reshape(n_days, n_stocks, n_features)
            # 转置为 (日期, 特征, 股票) 格式
            values = values.transpose(0, 2, 1)
        else:
            # 数据不完整，需要填充NaN或使用其他方法
            # 创建完整的数据框架
            from itertools import product
            full_index = pd.MultiIndex.from_product(
                [dates, stocks], 
                names=['datetime', 'instrument']
            )
            df_full = df.reindex(full_index)
            
            # 现在reshape
            values = df_full.values.reshape(n_days, n_stocks, n_features)
            values = values.transpose(0, 2, 1)
        
        return torch.tensor(values, dtype=torch.float, device=self.device), dates, stocks

    
    def __getitem__(self, slc: slice) -> "StockData":
        "Get a subview of the data given a date slice or an index slice."
        if slc.step is not None:
            raise ValueError("Only support slice with step=None")
        if isinstance(slc.start, str):
            return self[self.find_date_slice(slc.start, slc.stop)]
        start, stop = slc.start, slc.stop
        start = start if start is not None else 0
        stop = (stop if stop is not None else self.n_days) + self.max_future_days + self.max_backtrack_days
        start = max(0, start)
        stop = min(self.data.shape[0], stop)
        idx_range = slice(start, stop)
        data = self.data[idx_range]
        remaining = data.isnan().reshape(-1, data.shape[-1]).all(dim=0).logical_not().nonzero().flatten()
        data = data[:, :, remaining]
        return StockData(
            instrument=self._instrument,
            start_time=self._dates[start + self.max_backtrack_days].strftime("%Y-%m-%d"),
            end_time=self._dates[stop - 1 - + self.max_future_days].strftime("%Y-%m-%d"),
            max_backtrack_days=self.max_backtrack_days,
            max_future_days=self.max_future_days,
            features=self._features,
            device=self.device,
            preloaded_data=(data, self._dates[idx_range], self._stock_ids[remaining.tolist()])
        )

    def find_date_index(self, date: str, exclusive: bool = False) -> int:
        ts = pd.Timestamp(date)
        idx: int = self._dates.searchsorted(ts)  # type: ignore
        if exclusive and self._dates[idx] == ts:
            idx += 1
        idx -= self.max_backtrack_days
        if idx < 0 or idx > self.n_days:
            raise ValueError(f"Date {date} is out of range: available [{self._start_time}, {self._end_time}]")
        return idx
    
    def find_date_slice(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> slice:
        """
        Find a slice of indices corresponding to the given date range.
        For the input, both ends are inclusive. The output is a normal left-closed right-open slice.
        """
        start = None if start_time is None else self.find_date_index(start_time)
        stop = None if end_time is None else self.find_date_index(end_time, exclusive=False)
        return slice(start, stop)

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    @property
    def stock_ids(self) -> pd.Index:
        return self._stock_ids

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
            Parameters:
            - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
            a list of tensors of size `(n_days, n_stocks)`
            - `columns`: an optional list of column names
            """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
