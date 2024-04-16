import multiprocessing
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector

def roster_classifiers():
    classifiers = {
        'Logistic Regression': (LogisticRegression, {'random_state': 42, 'max_iter': 1000}),
        'Linear SVM': (SVC, {'kernel': "linear", 'C': 1, 'probability': True, 'random_state': 42}),
        # 'Decision Tree': (DecisionTreeClassifier, {'max_depth': 5, 'random_state': 42}),
        'Random Forest': (RandomForestClassifier, {'max_depth': 5, 'n_estimators': 10, 'random_state': 42}),
    }
    return classifiers

def evaluate_clf_perf(model_class, kwargs, list_features, df, impute_method='mean'):
    """ Given the model class, its arguments, the list of features to use, and the data,
    """
    aucs = []
    for fold_idx in df['fold_idx'].unique():
        X_train = df.loc[df['fold_idx']!=fold_idx, list_features].values
        y_train = df.loc[df['fold_idx']!=fold_idx, 'clf_label'].values
        X_test = df.loc[df['fold_idx']==fold_idx, list_features].values
        y_test = df.loc[df['fold_idx']==fold_idx, 'clf_label'].values
                
        # min-max normalization
        scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
        X_train = scaling.transform(X_train)
        X_test = scaling.transform(X_test)
                
        # impute missing values
        if impute_method == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif impute_method == 'median':
            imputer = SimpleImputer(strategy='median')
        elif impute_method == 'IterativeImputer':
            imputer = IterativeImputer(max_iter=10, random_state=0)
        elif impute_method == 'nearest':
            imputer = KNNImputer(n_neighbors=2, weights="uniform")
        else:
            raise ValueError(f'Imputation method {impute_method} not implemented')
        imputer.fit(X_train)
        X_train = imputer.transform(X_train)
        X_test = imputer.transform(X_test)

        # classification
        clf = model_class(**kwargs)
        clf.fit(X_train, y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        aucs.append(auc)
    auc_mean = sum(aucs) / len(aucs)
    auc_std = (sum([(x - auc_mean)**2 for x in aucs])/len(aucs))**0.5
    
    return auc_mean, auc_std