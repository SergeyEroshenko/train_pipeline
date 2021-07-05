import os
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import precision_score, f1_score, classification_report
from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count
from skopt.utils import use_named_args
from skopt import gp_minimize
import warnings
from mltrading.utils import Reader, Representation, Slicer
from configs import *


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # Initialize objects
    model = CatBoostClassifier
    cv = TimeSeriesSplit(n_splits=cv_splits)
    reader = Reader(data_path, symbol)
    slicer = Slicer(slice_by, window_size, step, take_profit, stop_loss, label_windows_size)
    stat_repr = Representation()

    # Select device for train
    device_count = get_gpu_device_count()
    if device_count == 0:
        task_type, devices = None, None
    else:
        task_type = "GPU"
        devices = ":".join([str(i) for i in range(device_count)])

    # Data reading and preparing. Dataset creating.
    data = reader.get_data()
    slicer.convert(data)
    windows = slicer.windows
    X = stat_repr.convert(windows).to_numpy()
    y = slicer.labels
    results = slicer.results

    X, X_test, y, y_test, _, res_test = train_test_split(
        X, y, results, test_size=test_split_size, random_state=random_seed, shuffle=False
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

            clf = model(
                **params,
                random_state=random_seed,
                verbose=False,
                thread_count=n_workers,
                task_type=task_type,
                devices=devices
                )
            clf.fit(X_train, y_train)
            y_pred_train = clf.predict(X_train)
            train_score = f1_score(y_train, y_pred_train, average='weighted')

            y_pred = clf.predict(X_val)
            val_score = f1_score(y_val, y_pred, average='weighted')
            prec_buy = precision_score(y_val, y_pred, labels=[1], average="micro")
            prec_sell = precision_score(y_val, y_pred, labels=[2], average="micro")

            precision_buy = np.append(precision_buy, prec_buy)
            precision_sell = np.append(precision_sell, prec_sell)
            print(f"Train score{idx+1} = {train_score:.5f}")
            print(f"Val score{idx+1} = {val_score:.5f}")
            # print("\n", classification_report(y_val, y_pred), "\n")

        x1 = precision_buy.mean()
        x2 = precision_sell.mean()
        x3 = precision_buy.std()
        x4 = precision_sell.std()

        total_score = 4 / (1 / x1 + 1 / x2 + 1 / (1 - x3) + 1 / (1 - x4))
        print(
            "Mean prec. buy: %.4f, mean prec. sell: %.4f,\nstd prec. buy: %4f, std prec. sell: %.4f" 
            % (x1, x2, x3, x4)
            )
        print("Score on current iteration: %.5f" % total_score)
        return -total_score

    result = gp_minimize(objective, space, n_calls=n_calls, random_state=random_seed, verbose=True)
    print('Best Score: %.3f' % (-result.fun))
    print('Best Parameters: %.5f' % (result.x))

    print("\n\nTrain model for testing best parameters.")
    # Model testing
    model_params = dict(zip(search_params, result.x))
    clf = model(
        **model_params,
        random_state=random_seed,
        verbose=False,
        thread_count=n_workers,
        task_type=task_type,
        devices=devices
        )
    clf.fit(X, y)
    clf.save_model(os.path.join(model_save_path, "model.cbm"))
    y_pred = clf.predict(X_test)[:, 0]
    score = classification_report(y_test, y_pred)
    print(f"\nTest score:\n{score}")

    positive = (y_pred==1) & (y_test==1) | (y_pred==1) & (y_pred==0)
    negative = (y_pred==2) & (y_test==2) | (y_pred==2) & (y_test==0)

    print("Profit in quote currency: %.5f" % res_test[positive].sum() - res_test[negative].sum())
    print("Total trades: %d" % positive.sum() + negative.sum())
