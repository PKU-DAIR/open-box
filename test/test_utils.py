import os
import pandas as pd
import numpy as np
import contextlib
import time
import traceback


seeds = [4774, 3711, 7238, 3203, 4254, 2137, 1188, 4356,  517, 5887,
         9082, 4702, 4801, 8242, 7391, 1893, 4400, 1192, 5553, 9039]


# timer tool
@contextlib.contextmanager
def timeit(name=''):
    print("[%s]Start." % name)
    start = time.time()
    yield
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print("[%s]Total time = %d hours, %d minutes, %d seconds." % (name, h, m, s))


def check_datasets(datasets, data_dir):
    for _dataset in datasets:
        try:
            _ = load_data(_dataset, data_dir)
        except Exception as e:
            print('Dataset - %s load error' % (_dataset))
            print(traceback.format_exc())
            raise


def load_data(dataset, data_dir):
    """
    no label encoding...
    """
    data_path = os.path.join(data_dir, "%s.csv" % dataset)

    # Load train data.
    if dataset in ['higgs', 'amazon_employee', 'spectf', 'usps', 'vehicle_sensIT', 'codrna']:
        label_col = 0
    elif dataset in ['rmftsa_sleepdata(1)']:
        label_col = 1
    else:
        label_col = -1

    if dataset in ['spambase', 'messidor_features']:
        header = None
    else:
        header = 'infer'

    if dataset in ['winequality_white', 'winequality_red']:
        sep = ';'
    else:
        sep = ','

    na_values = ["n/a", "na", "--", "-", "?"]
    keep_default_na = True
    df = pd.read_csv(data_path, keep_default_na=keep_default_na,
                     na_values=na_values, header=header, sep=sep)

    # Drop the row with all NaNs.
    df.dropna(how='all')

    # Clean the data where the label columns have nans.
    columns_missed = df.columns[df.isnull().any()].tolist()

    label_colname = df.columns[label_col]

    if label_colname in columns_missed:
        labels = df[label_colname].values
        row_idx = [idx for idx, val in enumerate(labels) if np.isnan(val)]
        # Delete the row with NaN label.
        df.drop(df.index[row_idx], inplace=True)

    train_y = df[label_colname].values

    # Delete the label column.
    df.drop(label_colname, axis=1, inplace=True)

    train_X = df
    return train_X, train_y
