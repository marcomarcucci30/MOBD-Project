import os
import warnings

import pandas as pd
from timeit import default_timer as timer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression

from classifier import best_classifier
from balancer import smote_tomek, svm_smote
from scaler import delete_outliers, standard_scaler, max_abs_scaler

from joblib import dump
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

training_path = './training_set.csv'
n_feat = 20


def fill_na(training, testing):
    """Substitutes missing values with the mean value calculated only in training set for each feature

    Args:
        training: training set
        testing: testing set

    Returns:
        training set without missing values
    """
    features = pd.read_csv(training_path, nrows=1).columns.tolist()[:n_feat]
    for feat in features:
        feature_mean = training[feat].mean()
        training[feat] = training[feat].fillna(feature_mean)
        if testing is not None:
            testing[feat] = testing[feat].fillna(feature_mean)
    return training, testing


def main():
    # retrieve training set
    training = pd.read_csv(training_path)

    # fill missing values
    training, _ = fill_na(training, None)

    # delete outliers
    training = delete_outliers(training)

    train_x = training.iloc[:, 0:n_feat].values
    train_y = training.iloc[:, n_feat].values

    # scaling data
    train_x, _ = max_abs_scaler(train_x, None)

    # balancing data
    train_x, train_y = svm_smote(train_x, train_y)

    # features selection
    k_best = SelectKBest(mutual_info_classif, k=15)
    train_x = k_best.fit_transform(train_x, train_y)

    start_classifier = timer()

    best_classifier_ = best_classifier(train_x, train_y, n_folds=10, metric='f1_macro')

    print("Elapsed time: ", timer()-start_classifier)

    dump(best_classifier_.best_estimator_, 'best_classifier.joblib')


if __name__ == "__main__":
    main()