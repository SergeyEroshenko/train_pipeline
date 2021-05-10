from mltrading.utils import Reader, Representation, Slicer
from sklearn.model_selection import cross_val_score ,TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
import warnings


warnings.filterwarnings('ignore')
multiply = 1e6
point = 1e-2
window_size = 10 * multiply
step = 2 * multiply
take_profit = 10000 * point
stop_loss = 5000 * point
label_windows_size = window_size
model = GradientBoostingClassifier
cv = TimeSeriesSplit(n_splits=5)

data_path = "/Users/sergeyerosenko/cryptoex/data"

reader = Reader(data_path, "BTCUSD")
slicer = Slicer("money", window_size, step, take_profit, stop_loss, label_windows_size)
stat_repr = Representation()

data = reader.get_data()
slicer.convert(data)
X = stat_repr.convert(slicer.windows).to_numpy()
y = slicer.labels
print(f"Dataset parameters: num windows {X.shape[0]}, num features {X.shape[1]}.")

for idx, (train_idx, val_idx) in enumerate(cv.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    clf = model(random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    score = f1_score(y_val, y_pred, average='micro')
    print(f"score{idx+1} = {score:.5f}")