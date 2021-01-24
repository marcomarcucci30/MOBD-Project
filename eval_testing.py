import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression

from balancer import svm_smote
from scaler import delete_outliers, standard_scaler, max_abs_scaler
from joblib import load
from training_best_configuration import fill_na
from hyperparameter_tuning import evaluate_classifier

training_path = './training_set.csv'
n_feat = 20


def read_config(path):
    """Read testing set path from config.txt

    Args:
        path: path file config.txt

    Returns: path testing set

    """
    f = open(path, 'r')
    path_test = f.readline().replace('\n', '')
    f.close()
    return path_test


def main():
    path_test = read_config('./config.txt')
    training = pd.read_csv(training_path)
    testing = pd.read_csv(path_test)

    training, testing = fill_na(training, testing)

    training = delete_outliers(training)

    train_x = training.iloc[:, 0:n_feat].values
    train_y = training.iloc[:, n_feat].values
    test_x = testing.iloc[:, 0:n_feat].values
    test_y = testing.iloc[:, n_feat].values

    train_x, test_x = max_abs_scaler(train_x, test_x)
    train_x, train_y = svm_smote(train_x, train_y)  # this line has been added
    k_best = SelectKBest(mutual_info_classif, k=15)
    k_best.fit_transform(train_x, train_y)  # only for get_support
    mask = k_best.get_support()
    test_x = test_x[:, mask]

    clf = load('best_classifier.joblib')
    f1_score = evaluate_classifier(clf, test_x, test_y)
    print("F1_MACRO score metric:", f1_score)


if __name__ == "__main__":
    main()
