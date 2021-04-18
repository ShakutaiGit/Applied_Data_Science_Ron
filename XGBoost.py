from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split, cross_val_score
import tqdm as tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from xgboost import XGBClassifier
import warnings



class xgboost:
    def __init__(self, df, prec):
        self.df = df
        self.split_prec = prec

    def split_to_train_test(self):
        y = self.df["SARS-Cov-2 exam result"]
        X = self.df.drop(["SARS-Cov-2 exam result"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, shuffle=True)
        X_train.reset_index(inplace=True, drop=True)
        cv_outer = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
        max_score = 0
        best_parameters = {'max_depth': {}, 'n_estimators': {}, 'learning_rate':{}}
        for train_idx, val_idx in tqdm.tqdm(cv_outer.split(X_train, y_train)):
            train_data, val_data = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train.reset_index(inplace=True, drop=True)
            train_target, val_target = y_train[train_idx], y_train[val_idx]
            estimator = XGBClassifier(
                objective='binary:logistic',
                nthread=4,
                seed=42
            )
            cv_inner = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
            params = {
                'max_depth': [2, 4, 8, 16, 32, 64],
                'n_estimators': [10, 20, 30, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
                'learning_rate': [0.1, 0.01, 0.05]
            }
            # print(model.get_params().keys())
            gd_search = GridSearchCV(estimator, params, scoring='roc_auc', n_jobs=-1, cv=cv_inner).fit(train_data, train_target)
            best_model = gd_search.best_estimator_
            classifier = best_model.fit(train_data, train_target)
            y_pred_prob = classifier.predict_proba(val_data)[:, 1]
            auc = metrics.roc_auc_score(val_target, y_pred_prob)
            print("Val Acc:", auc, "Best GS Acc:", gd_search.best_score_, "Best Params:", gd_search.best_params_)
            self.update_the_best_parameter(best_parameters, gd_search.best_params_)

        print(best_parameters)
        mean_best_parameters = self.get_top_values_dict(best_parameters)
        print(X_train)
        model = XGBClassifier(max_depth=mean_best_parameters['max_depth'], n_estimators=mean_best_parameters['n_estimators'], learning_rate=mean_best_parameters['learning_rate']).fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        print("AUC", metrics.roc_auc_score(y_test, y_pred_prob))
        # auc = metrics.f1_score(y_test, y_pred_prob)
        # print(auc)
        # print(metrics.confusion_matrix(y_test, y_pred_prob))

    def update_the_best_parameter(self,best_parameter,iter):
        for key in iter.keys():
            value =iter[key]
            if value in best_parameter[key]:
                best_parameter[key][value]+=1
            else:
                best_parameter[key][value]=1

    def get_top_values_dict(self,best_parameter):
        res = {}
        for key in best_parameter.keys():
            max_val = max(best_parameter[key].values())
            for key1 in best_parameter[key]:
                if best_parameter[key][key1] == max_val:
                    res[key] = key1
        return res
