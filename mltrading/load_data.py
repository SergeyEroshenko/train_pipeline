import os
from glob import glob
import pandas as pd
import numpy as np


class Reader:

    def __init__(self, data_path: str, symbol: str):
        self.data_path = data_path
        self.symbol = symbol
        self.data = list()

    def get_data(self):
        files_list = sorted(
            glob(os.path.join(self.data_path, self.symbol, '*.csv'))
            )

        for file in files_list:
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
        data['time_diff'] = data['timestamp'].diff()
        data['price_diff'] = data['price'].pct_change().fillna(0)
        data['codirect'] = (
            np.sign(data['price_diff']) == np.sign(data['side'])
            ).astype(int)
        data = data.set_index('traded')
        return data


class Slicer:

    def __init__(self, by: str, window_size: int, step: int) -> None:
        if by not in ['time', 'tick', 'quantity', 'money']:
            raise ValueError("by not in: 'time', 'tick', 'quantity', 'money'")

        self.data = list()
        self.by = by
        self.window_size = window_size
        self.step = step

    def convert(self, data):
        if self.by == 'money':
            data = self.money_index(data)

        windows_amount = (int(data.index.max()) - self.window_size)//self.step
        for i in range(windows_amount):
            start_value = i * self.step
            end_value = i * self.step + self.window_size
            window = data[
                (data.index >= start_value)
                & (data.index < end_value)
                ]
            self.data.append(window)

    @staticmethod
    def money_index(data):
        data.index = data['quantity'] * data['price']
        return data

