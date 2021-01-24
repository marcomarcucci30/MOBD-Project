import os
import warnings

import numpy as np
import pandas as pd

import sklearn
import sklearn.metrics as metrics
from timeit import default_timer as timer
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, GenericUnivariateSelect

from scaler import minMaxScaler, robustScale, max_abs_scaler, standard_scaler, delete_outliers
from balancer import undersampling, smote_nc, smote_tomek, svm_smote, smote, over_sample, adasyn

from classifier import qda_param_selection, k_neighbors_param_selection, perceptron_adaboost_param_selection, \
    gbc_param_selection

import seaborn as sns
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"



#import matplotlib
# import tensorflow as tf
# from matplotlib import pyplot as plt
# from numba import jit, cuda


print("numpy:", np.__version__)
print("pandas:", pd.__version__)
#conda print("matplotlib:", matplotlib.__version__)
print('scikit-learn:', sklearn.__version__)


"""Dictionary of Features Selection index tested"""
feat_model = {
    'mutual_info_classif': mutual_info_classif,
    'chi': chi2,
    'f_classif': f_classif,
    'none': None
}

"""Dictionary of balance technique tested"""
balance_techniques = {
    'smote_tomek': smote_tomek,
    'undersampling': undersampling,
    'adasyn': adasyn,
    'svm_smote': svm_smote,
    'smote': smote,
    'smote_nc': smote_nc,
    'over_sample': over_sample,
    'none': None
}

"""Dictionary of scale technique tested"""
scale_techniques = {
    'max_abs_scaler': max_abs_scaler,
    'standard_scaler': standard_scaler,
    'minMaxScaler': minMaxScaler,
    'robustScale': robustScale,
    'none': None,
}

"""Manage outliers"""
outliers = {
    'none': None,
    'delete': delete_outliers
}


def get_na_count(dataset):
    """Function which count the number of missing values for each attributes

    Args:
        dataset: dataset

    Returns:
        number of missing values for each attributes
    """
    boolean_mask = dataset.isna()
    return boolean_mask.sum(axis=0)


def fill_na(dataset):
    """Substitutes missing values with the mean value calculated only in training set for each feature

    Args:
        dataset: dataset

    Returns:
        dataset without missing values
    """
    x = dataset.sample(frac=0.80, random_state=0)
    features = pd.read_csv(proj_name, nrows=1).columns.tolist()[:n_feat]
    for feat in features:
        feature_mean = x[feat].mean()
        dataset[feat] = dataset[feat].fillna(feature_mean)
    return dataset


def main():
    global proj_name, n_feat
    start = timer()
    proj_name = './training_set.csv'
    n_feat = 20

    for outlier in outliers:

        for k in [15]:

            for mod in feat_model:

                for scaler in scale_techniques:

                    for balancer in balance_techniques:

                        if mod == 'chi' and scaler != 'minMaxScaler':
                            break

                        global x, y

                        # Retrieve dataset
                        dataset = pd.read_csv(proj_name)
                        # print('Number of NaN Values: \n' + str(get_na_count(dataset)))

                        # check proportion class
                        count = dataset['CLASS'].value_counts()
                        c1 = count[0]
                        c2 = count[1]
                        c3 = count[2]
                        c4 = count[3]
                        ct = c1 + c2 + c3 + c4
                        # print('CLASS 1 percentage:',c1*100/ct, '%\nCLASS 2 percentage:',c2*100/ct,
                        #     '%\nCLASS 3 percentage:',c3*100/ct, '%\nCLASS 4 percentage:',c4*100/ct,'%')

                        # Fill missing values on dataset
                        dataset = fill_na(dataset)

                        # Split Dataset in training set and testing set
                        training = dataset.sample(frac=0.80, random_state=0)
                        testing = dataset.drop(training.index)

                        # ************OUTLIER************
                        if outlier != 'none':
                            training = outliers[outlier](training)

                        train_x = training.iloc[:, 0:n_feat].values
                        train_y = training.iloc[:, n_feat].values

                        test_x = testing.iloc[:, 0:n_feat].values
                        test_y = testing.iloc[:, n_feat].values


                        # ************BALANCING************
                        if balancer != 'none':
                            train_x, train_y = balance_techniques[balancer](train_x, train_y)

                        # ************SCALING************
                        if scaler != 'none':
                            train_x, test_x = scale_techniques[scaler](train_x, test_x)

                        # ************FEATURES SELECTION************
                        if mod != 'none':
                            feat_val = feat_model[mod]

                            k_best = SelectKBest(feat_val, k=k)

                            train_x = k_best.fit_transform(train_x, train_y)
                            mask = k_best.get_support()

                            # Features selection in testing set too
                            test_x = test_x[:, mask]

                            #for i in range(len(k_best.scores_)):
                            #   print('Feature %d: %f' % (i, k_best.scores_[i]))

                        start_classifier = timer()

                        # svm_classifier = svm_param_selection(train_x, train_y, n_folds=10, metric='f1_macro')

                        # mlp_classifier = mlp_param_selection(train_x, train_y, n_folds=10, metric='f1_macro')
                        # mlp_first_paramgrid_classifier = mlp_param_selection_first_paramgrid(train_x, train_y, n_folds=10,metric='f1_macro')
                        # mlp_second_paramgrid_classifier = mlp_param_selection_second_paramgrid(train_x, train_y,n_folds=10,metric='f1_macro')
                        # mlp_third_paramgrid_classifier = mlp_param_selection_second_paramgrid(train_x, train_y,n_folds=10,metric='f1_macro')

                        # rf_classifier = random_forest_param_selection(train_x, train_y, n_folds=10, metric='f1_macro')

                        qda_classifiers = qda_param_selection(train_x, train_y, n_folds=10, metric='f1_macro')
                        # lda_shrink_classifiers = lda_shrink_param_selection(train_x, train_y, n_folds=10, metric='f1_macro')
                        # lda_svd_classifiers = lda_svd_param_selection(train_x, train_y, n_folds=10, metric='f1_macro')

                        #gbc_classifiers = gbc_param_selection(train_x, train_y, n_folds=10, metric='f1_macro')

                        #k_neighbors_classifiers = k_neighbors_param_selection(train_x, train_y, n_folds=10, metric='f1_macro')
                        #perceptron_adaboost_classifier = perceptron_adaboost_param_selection(train_x, train_y, n_folds=10, metric='f1_macro')

                        f1_score = evaluate_classifier(qda_classifiers, test_x, test_y)

                        print(f1_score, k, mod, scaler, balancer, outlier, qda_classifiers.best_params_,
                              str(-start_classifier + timer()), flush=True)

                        # UPDATES FILE
                        results = open('results.txt', 'a')
                        out = ' '.join([str(k), mod, str(f1_score), scaler, balancer, outlier,
                                        str(qda_classifiers.best_params_), str(-start_classifier + timer()), '\n'])
                        results.write(out)
                        results.close()

            print("Elapsed time:", timer() - start)


def evaluate_classifier(classifier, test_x, test_y):
    """Evaluate the classifier on the testing set specified

    Args:
        classifier: Classifier to evaluate
        test_x: Testing set without Class Target
        test_y: Testing set Class Target

    Returns:
        F1_MACRO metric
    """
    pred_y = classifier.predict(test_x)
    f1_score = metrics.f1_score(test_y, pred_y, average='macro')
    return f1_score


if __name__ == '__main__':
    main()
