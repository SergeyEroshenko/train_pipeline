from re import search
import numpy as np
from mltrading.utils import Reader, Representation, Slicer
from sklearn.model_selection import cross_val_score ,TimeSeriesSplit, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, f1_score, classification_report
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
import warnings


warnings.filterwarnings('ignore')
# Initialize parameters and objects
data_path = "/Users/sergeyerosenko/cryptoex/data"

multiply = 1e6
point = 1e-2
window_size = 10 * multiply
step = 2 * multiply
take_profit = 10000 * point
stop_loss = 10000 * point
label_windows_size = window_size

search_params = ['max_depth', 'n_estimators', 'learning_rate', 'min_samples_split', 'min_samples_leaf']

space = [
    Integer(1, 7, name=search_params[0]),
    Integer(5, 5000, name=search_params[1]),
    Real(10**-5, 10**5, 'log-uniform', name=search_params[2]),
    Integer(2, 6, name=search_params[3]),
    Integer(1, 6, name=search_params[4])
]

model = GradientBoostingClassifier
cv = TimeSeriesSplit(n_splits=5)
reader = Reader(data_path, "BTCUSD")
slicer = Slicer("money", window_size, step, take_profit, stop_loss, label_windows_size)
stat_repr = Representation()
# Data reading and preparing. Dataset creating.
data = reader.get_data()
slicer.convert(data)

X = stat_repr.convert(slicer.windows).to_numpy()
y = slicer.labels
results = slicer.results

X, X_test, y, y_test, _, res_test = train_test_split(
    X, y, results, test_size=0.2, random_state=1, shuffle=False
    )

num_intersect = np.floor(window_size / step).astype(int)
X = X[: -num_intersect]
y = y[: -num_intersect]
print(f"Dataset parameters: num windows {X.shape[0]}, num features {X.shape[1]}.")
# Finding optimal model parameters with cross validation apply.
@use_named_args(space)
def objective(**params):
    print(params)
    precision_buy = np.empty(0)
    precision_sell = np.empty(0)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train = X_train[: -num_intersect]
        y_train = y_train[: -num_intersect]

        clf = model(random_state=1, **params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        val_score = f1_score(y_val, y_pred, average='weighted')
        prec_buy = precision_score(y_val, y_pred, labels=[1], average="micro")
        prec_sell = precision_score(y_val, y_pred, labels=[2], average="micro")

        precision_buy = np.append(precision_buy, prec_buy)
        precision_sell = np.append(precision_sell, prec_sell)
        print(f"Val score{idx+1} = {val_score:.5f}")
        print("\n", classification_report(y_val, y_pred), "\n")

    x1 = precision_buy.mean()
    x2 = precision_sell.mean()
    x3 = precision_buy.std()
    x4 = precision_sell.std()

    total_score = 4 / (1 / x1 + 1 / x2 + 1 / (1 - x3) + 1 / (1 - x4))
    print(x1, x2, x3, x4)
    print(-total_score)
    return -total_score

result = gp_minimize(objective, space, n_calls=20, random_state=1, verbose=True)
print('Best Score: %.3f' % (result.fun))
print('Best Parameters: %s' % (result.x))

# Model testing
model_params = dict(zip(search_params, result.x))
clf = model(random_state=1)
clf.fit(X, y)
y_pred = clf.predict(X_test)
score = classification_report(y_test, y_pred)
print(f"\n\nTest score:\n{score}")

positive = (y_pred==1) & (y_test==1) | (y_pred==1) & (y_pred==0)
negative = (y_pred==2) & (y_test==2) | (y_pred==2) & (y_test==0)

print(res_test[positive].sum() - res_test[negative].sum())