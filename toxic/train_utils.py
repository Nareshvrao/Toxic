from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from toxic.rocauccallback import RocAucEval
import time

import numpy as np


def _train_model(model, batch_size, train_x, train_y, val_x, val_y, use_roc):
    best_loss = -1
    best_roc_auc = -1
    best_weights = None
    best_epoch = 0

    current_epoch = 0

    # RocAuc = RocAucEval(validation_data=(val_x, val_y), interval=1)

    while True:
        model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
        # model.fit(train_x, train_y, batch_size=batch_size, epochs=1, validation_data=(val_x, val_y), callbacks=[RocAuc])
        y_pred = model.predict(val_x, batch_size=batch_size * 2, verbose=1)

        total_loss = 0
        for j in range(6):
            loss = log_loss(val_y[:, j], y_pred[:, j])
            total_loss += loss

        total_loss /= 6.

        total_roc_auc_score = 0
        for j in range(6):
            roc_auc_loss = roc_auc_score(val_y[:, j], y_pred[:, j])
            total_roc_auc_score += roc_auc_loss

        total_roc_auc_score /= 6.

        print("Epoch {0} log_loss {1} ROC_AUC {2} best_log_loss {3} best_ROC_AUC {4}".format(current_epoch, total_loss,
                                                                                             total_roc_auc_score,
                                                                                             best_loss, best_roc_auc))

        current_epoch += 1
        if use_roc == "True":
            # for roc auc score
            if total_roc_auc_score > best_roc_auc or best_roc_auc == -1:
                best_roc_auc = total_roc_auc_score
                best_weights = model.get_weights()
                best_epoch = current_epoch
            else:
                if current_epoch - best_epoch == 5:
                    break

            if total_loss < best_loss or best_loss == -1:
                best_loss = total_loss
                # best_weights = model.get_weights()
                # best_epoch = current_epoch
            # else:
            #     if current_epoch - best_epoch == 5:
            #         break
        else:
            if total_loss < best_loss or best_loss == -1:
                best_loss = total_loss
                best_weights = model.get_weights()
                best_epoch = current_epoch
            else:
                if current_epoch - best_epoch == 5:
                    break

            # for roc auc score
            if total_roc_auc_score > best_roc_auc or best_roc_auc == -1:
                best_roc_auc = total_roc_auc_score
            #     best_weights = model.get_weights()
            #     best_epoch = current_epoch
            # else:
            #     if current_epoch - best_epoch == 5:
            #         break

    model.set_weights(best_weights)
    return model


def train_folds(X, y, fold_count, batch_size, get_model_func, cv, use_roc):
    start = time.time()
    if cv == "True":
        fold_size = len(X) // fold_count
        models = []
        for fold_id in range(0, fold_count):
            start = time.time()
            print('#' * 50)
            print('Fold {} started'.format(fold_id))
            fold_start = fold_size * fold_id
            fold_end = fold_start + fold_size

            if fold_id == fold_size - 1:
                fold_end = len(X)

            train_x = np.concatenate([X[:fold_start], X[fold_end:]])
            train_y = np.concatenate([y[:fold_start], y[fold_end:]])

            val_x = X[fold_start:fold_end]
            val_y = y[fold_start:fold_end]

            model = _train_model(get_model_func(), batch_size, train_x, train_y, val_x, val_y, use_roc)
            models.append(model)
            total_time = time.time() - start
            mins, sec = divmod(total_time, 60)
            hrs, mins = divmod(mins, 60)
            print('Time taken : {:.0f}h {:.0f}m {:.0f}s'.format(hrs, mins, sec))
    else:
        print('No Cross Validation')
        train_x, val_x, train_y, val_y = train_test_split(X, y, train_size=0.9, random_state=233)
        models = _train_model(get_model_func(), batch_size, train_x, train_y, val_x, val_y, use_roc)
        total_time = time.time() - start
        mins, sec = divmod(total_time, 60)
        hrs, mins = divmod(mins, 60)
        print('Time taken : {:.0f}h {:.0f}m {:.0f}s'.format(hrs, mins, sec))

    return models
