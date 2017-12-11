#!/usr/bin/env python

from argparse import ArgumentParser
import logging

from pandas import read_csv
from sklearn.model_selection import PredefinedSplit
from sklearn.utils import safe_indexing
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

index_col = 'Unnamed: 0'
text_cols = ['description', 'designation']
enum_cols = ['country', 'province', 'region_1', 'region_2', 'variety', 'winery']
num_cols = ['price']
y_col = 'points'
fold_pattern = [-1, -1, 0, -1, -1]  # Test index: [2, 7, 12, 17, ...]
prediction_divide = 90  # Prediction is right if both predicted and actual value are at the same side from it


def read_dataset(filename):
    dataset = read_csv(filename, header=0, index_col=index_col)
    for col in text_cols:
        del dataset[col]
    for col in enum_cols:
        mapping = {name: i for i, name in enumerate(dataset[col].unique())}
        dataset[col] = dataset[col].map(mapping).fillna('')
    for col in num_cols:
        dataset[col] = dataset[col].fillna(dataset[col].mean())
    return dataset


def split_dataset(dataset):
    X = dataset.drop(y_col, axis=1)
    y = dataset[y_col]
    test_fold = (fold_pattern * ((dataset.shape[0] - 1) // len(fold_pattern) + 1))[:dataset.shape[0]]
    splitter = PredefinedSplit(test_fold)
    for train_index, test_index in splitter.split():
        X_train, X_test = safe_indexing(X, train_index), safe_indexing(X, test_index)
        y_train, y_test = safe_indexing(y, train_index), safe_indexing(y, test_index)
    return X_train, y_train, X_test, y_test


def is_predicted_right(predicted, actual):
    return (actual > prediction_divide) == (predicted > prediction_divide)


def prettify(d):
    return '\n'.join(['{:8s}: {:4.1f}%'.format(k, 100 * v) for k, v in d.items()])


def main():
    parser = ArgumentParser()
    parser.add_argument('filename', help='path to dataset.csv (default: STDIN).', default='/dev/stdin')
    filename = parser.parse_args().filename

    dataset = read_dataset(filename)
    X_train, y_train, X_test, y_test = split_dataset(dataset)
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    predictions = dtc.predict(X_test)

    logger.info(prettify({'Accuracy score': accuracy_score(y_test, predictions)}))
    logger.info('Column importances:\n{}'.format(prettify(dict(zip(X_train.keys(), dtc.feature_importances_)))))
    n_predicted_right = sum([is_predicted_right(predicted, actual) for predicted, actual in zip(predictions, y_test)])
    n_tested = len(predictions)
    logger.info('Predicted right: {} / {} ({:.1f}%)'.format(
        n_predicted_right,
        n_tested,
        100 * n_predicted_right / n_tested
    ))


if __name__ == '__main__':
    main()
