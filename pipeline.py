from mltrading.load_data import Reader, Representation, Slicer


multiply = 1e6
point = 1e-2
window_size = 100 * multiply
step = 50 * multiply
take_profit = 100000 * point
stop_loss = 50000 * point
label_windows_size = window_size

data_path = "/Users/sergeyerosenko/cryptoex/data"

reader = Reader(data_path, "BTCUSD")
slicer = Slicer("money", window_size, step, take_profit, stop_loss, label_windows_size)
stat_repr = Representation()
data = reader.get_data()
slicer.convert(data)
slicer.windows = stat_repr.convert(slicer.windows)


print(slicer.labels)
print(slicer.results)
print(slicer.windows)
