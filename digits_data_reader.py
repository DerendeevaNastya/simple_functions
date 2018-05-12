import pandas as pd


def get_data_from_file(filename):
    data = pd.read_csv(filename).as_matrix()
    xtrain = data[:, 1:]
    train_label = data[:, 0]
    print(train_label)
    print(xtrain.shape)
    print(train_label.shape)
    return data
