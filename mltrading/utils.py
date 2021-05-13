import os
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings


class Reader:

    def __init__(self, data_path: str, symbol: str):
        self.data_path = data_path
        self.symbol = symbol
        self.data = list()

    def get_data(self):
        files_list = sorted(
            glob(os.path.join(self.data_path, self.symbol, '*.csv'))
            )

        for file in tqdm(files_list, total=len(files_list), desc='Data Loading: '):
            df = pd.read_csv(file, sep=';')
            self.data.append(df)

        self.data = pd.concat(self.data)

        self.data['side'] = self.data['side'].map({'buy': 1, 'sell': -1})
        self.data['timestamp'] = self.data['timestamp']\
            .astype('datetime64[ns]')
        self.data = self.data.sort_values('timestamp')
        self.data = self.data.reset_index(drop=True)
        return self.data


class Representation:

    def convert(self, data):
        converted_data = list()
        settings_efficient = settings.MinimalFCParameters()

        for idx, window in enumerate(data):
            window['time_diff'] = window['timestamp'].diff()
            window['time_diff'] = window['time_diff'].dt.total_seconds()
            window['price_diff'] = window['price'].pct_change().fillna(0)
            window['codirect'] = (
                np.sign(window['price_diff']) == np.sign(window['side'])
                ).astype(int)
            window = window.drop(['price', 'timestamp'], axis=1)
            window.loc[:, 'id'] = idx
            converted_data.append(window)
        converted_data = pd.concat(converted_data).reset_index(drop=True)
        converted_data['time_diff'] = converted_data['time_diff'].fillna(0)
        converted_data = extract_features(
            converted_data, 
            column_id='id', 
            impute_function=impute, 
            default_fc_parameters=settings_efficient
            )

        return converted_data


class Slicer:

    def __init__(
        self, by: str,
        window_size: int,
        step: int,
        take_profit: int,
        stop_loss: int,
        label_windows_size: int
        ):
        if by not in ['time', 'tick', 'quantity', 'money']:
            raise ValueError("'by' not in: ['time', 'tick', 'quantity', 'money']")

        self.windows = list()
        self.labels = np.empty(0, dtype=np.int)
        self.results = np.empty(0, dtype=np.float)
        self.by = by
        self.window_size = window_size
        self.step = step
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.label_windows_size = label_windows_size

    def convert(self, data):
        if self.by == 'money':
            data = self.money_index(data)

        windows_amount = (data.index.max() - (self.window_size + self.label_windows_size))//self.step
        windows_amount = int(windows_amount)

        for i in tqdm(range(windows_amount), total=windows_amount, desc="Slicing: "):
            start_value = i * self.step
            end_value = i * self.step + self.window_size
            window = data[
                (data.index >= start_value)
                & (data.index < end_value)
                ]
            label_data = data[
                (data.index >= end_value)
                & (data.index < end_value + self.label_windows_size)
                ]
            label, result = self.check_label(label_data)
            self.windows.append(window)
            self.labels = np.append(self.labels, label)
            self.results = np.append(self.results, result)

    def check_label(self, data: pd.DataFrame):
        data['price'] = data['price'] - data['price'].iloc[0]
        prices = data['price'].values
        label = int(0)
        result = prices[-1]
        last_index = data.shape[0] - 1
        tp_indexes = np.where(prices >= self.take_profit)[0]
        sl_indexes = np.where(prices <= -self.stop_loss)[0]
        if tp_indexes.size != 0:
            first_tp = tp_indexes[0]
        else:
            first_tp = last_index

        if sl_indexes.size != 0:
            first_sl = sl_indexes[0]
        else:
            first_sl = last_index

        if (first_tp < first_sl) & (first_tp < last_index):
            label = int(1)
            result = prices[first_tp]
        if (first_sl < first_tp) & (first_sl < last_index):
            label = int(2)
            result = prices[first_sl]

        return label, result

    @staticmethod
    def money_index(data):
        new_index = data['quantity'] * data['price']
        data.index = new_index.cumsum()
        return data

