import numpy as np
import matplotlib.pyplot as plt


def shuffle_data(data):
    X = data['X']
    t = data['t']
    permutation = np.random.permutation(len(t))
    return X[permutation], t[permutation]


def split_data(data, num_folds, fold):
    fold_size = int(len(data[0]) / num_folds)
    fold_begin = int((fold - 1) * fold_size)
    fold_end = fold_begin + fold_size
    data_fold = data[0][fold_begin: fold_end], data[1][fold_begin: fold_end]
    X_rest = np.concatenate((data[0][0: fold_begin], data[0][fold_end: len(data)]))
    t_rest = np.concatenate((data[1][0: fold_begin], data[1][fold_end: len(data)]))
    data_rest = X_rest, t_rest
    return data_fold, data_rest


def train_model(data, lambd):
    x = data[0]
    t = data[1]
    XTX = np.transpose(x).dot(x)
    return np.linalg.inv(XTX + lambd * np.identity(XTX.shape[0])).dot(np.transpose(x)).dot(t)


def predict(data, model):
    return data[0].dot(model)


def loss(data, model):
    prediction = predict(data, model)
    return (np.transpose(data[1] - prediction).dot(data[1] - prediction)) / len(data[0])


def cross_validation(data, num_folds, lambd_seq):
    data = shuffle_data(data)
    cv_error = []

    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0

        for fold in range(1, num_folds + 1):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error.append(cv_loss_lmd / num_folds)
    return cv_error


if __name__ == '__main__':

    data_train = {'X': np.genfromtxt('data_train_X.csv', delimiter=','),
                  't': np.genfromtxt('data_train_y.csv', delimiter=',')}

    data_test = {'X': np.genfromtxt('data_test_X.csv', delimiter=','),
                 't': np.genfromtxt('data_test_y.csv', delimiter=',')}

    # construct lambda values
    lambd_seq = np.linspace(0.02, 1.5, num=50)
    fold_5_cross_validation_error = cross_validation(data_train, 5, lambd_seq)
    fold_10_cross_validation_error = cross_validation(data_train, 10, lambd_seq)

    train_error = []
    test_error = []
    for lambd in lambd_seq:
        model = train_model((data_train["X"], data_train["t"]), lambd)
        train_error.append(loss((data_train["X"], data_train["t"]), model))
        test_error.append(loss((data_test["X"], data_test["t"]), model))

    plt.plot(lambd_seq, fold_5_cross_validation_error, label="5-fold-error")
    plt.plot(lambd_seq, fold_10_cross_validation_error, label="10-fold-error")
    plt.plot(lambd_seq, train_error, label="train_error")
    plt.plot(lambd_seq, test_error, label="test_error")
    plt.legend(loc="upper right")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Error")
    plt.savefig("error.png")
    plt.show()
