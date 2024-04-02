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
from sklearn.feature_selection import SelectFromModel

def roster_classifiers():
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Linear SVM': SVC(kernel="linear", C=1, probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(
            max_depth=5, n_estimators=10, random_state=42
        ),
    }
    return classifiers

def run_classification_experiments(data, feat_combo, classifiers, results_csv, num_folds=5, impute_method='mean'):
    # dataframe to store results
    results_detail = pd.DataFrame()
    results_simple = pd.DataFrame()
    
    # classification using different feature combinations
    for combo_name, list_features in feat_combo.items():
        row_detail = {'Features': [combo_name]}
        row_simple = {'Features': [combo_name]}

        for classifier_name, clf in tqdm(classifiers.items(), total=len(classifiers), desc=f'Classification: {combo_name}'):            
            accs, specs, senss, aucs = [], [], [], []
            for fold_idx in range(1, num_folds+1):
                X_train = data.loc[data['fold_idx']!=fold_idx, list_features].values
                y_train = data.loc[data['fold_idx']!=fold_idx, 'category'].values
                X_test = data.loc[data['fold_idx']==fold_idx, list_features].values
                y_test = data.loc[data['fold_idx']==fold_idx, 'category'].values
                
                # scaling
                scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
                X_train = scaling.transform(X_train)
                X_test = scaling.transform(X_test)
                
                # Impute missing values
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

                # feature selection
                lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
                model = SelectFromModel(lsvc, threshold='median', prefit=True)
                X_train = model.transform(X_train)
                X_test = model.transform(X_test)

                # classification
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                spec = tn / (tn + fp)
                sens = tp / (tp + fn)
                auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

                accs.append(acc)
                specs.append(spec)
                senss.append(sens)
                aucs.append(auc)
                
                row_detail[f'{classifier_name}_acc_{fold_idx}'] = [acc]
                row_detail[f'{classifier_name}_spec_{fold_idx}'] = [spec]
                row_detail[f'{classifier_name}_sens_{fold_idx}'] = [sens]
                row_detail[f'{classifier_name}_auc_{fold_idx}'] = [auc]

            acc_mean = sum(accs) / len(accs)
            acc_std = (sum([(x - acc_mean)**2 for x in accs])/len(accs))**0.5
            spec_mean = sum(specs) / len(specs)
            spec_std = (sum([(x - spec_mean)**2 for x in specs])/len(specs))**0.5
            sens_mean = sum(senss) / len(senss)
            sens_std = (sum([(x - sens_mean)**2 for x in senss])/len(senss))**0.5
            auc_mean = sum(aucs) / len(aucs)
            auc_std = (sum([(x - auc_mean)**2 for x in aucs])/len(aucs))**0.5

            row_simple[f'{classifier_name}_acc'] = [f"{acc_mean:.3f}±{acc_std:.3f}"]
            row_simple[f'{classifier_name}_spec'] = [f"{spec_mean:.3f}±{spec_std:.3f}"]
            row_simple[f'{classifier_name}_sens'] = [f"{sens_mean:.3f}±{sens_std:.3f}"]
            row_simple[f'{classifier_name}_auc'] = [f"{auc_mean:.3f}±{auc_std:.3f}"]

        row_detail = pd.DataFrame(row_detail)
        row_simple = pd.DataFrame(row_simple)
        results_detail = pd.concat([results_detail, row_detail], axis=0)
        results_simple = pd.concat([results_simple, row_simple], axis=0)

    results_detail.to_csv(results_csv, index=False)
    results_simple.to_excel(results_csv.replace('.csv', '_simple.xlsx'), index=False)
