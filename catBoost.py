from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split, cross_val_score
import tqdm as tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from xgboost import XGBClassifier


class catboost:
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
        best_parameters = {'iterations': {}, 'learning_rate':{}}
        for train_idx, val_idx in tqdm.tqdm(cv_outer.split(X_train, y_train)):
            train_data, val_data = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train.reset_index(inplace=True, drop=True)
            train_target, val_target = y_train[train_idx], y_train[val_idx]
            classifier = CatBoostClassifier()
            cv_inner = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
            params = {
                'iterations': [2, 4, 8],
                'learning_rate': [0.1, 0.01, 0.05]
            }
            # print(model.get_params().keys())

            gd_search = GridSearchCV(classifier, params, scoring='roc_auc', n_jobs=-1, cv=cv_inner).fit(train_data, train_target)
            best_model = gd_search.best_estimator_
            classifier = best_model.fit(train_data, train_target)
            y_pred_prob = classifier.predict_proba(val_data)[:, 1]
            auc = metrics.roc_auc_score(val_target, y_pred_prob)
            # print("Val Acc:", auc, "Best GS Acc:", gd_search.best_score_, "Best Params:", gd_search.best_params_)
            self.update_the_best_parameter(best_parameters, gd_search.best_params_)


        mean_best_parameters = self.get_top_values_dict(best_parameters)
        model = CatBoostClassifier(max_depth=mean_best_parameters['iterations'], learning_rate=mean_best_parameters['learning_rate']).fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        print("catboost")
        print("rocAUC", metrics.roc_auc_score(y_test, y_pred_prob))
        print("f1",metrics.f1_score(y_test, np.round(y_pred_prob)))
        print("accuracy_score", metrics.accuracy_score(y_test, np.round(y_pred_prob)))
        # print(metrics.classification_report(y_test, np.round(y_pred_prob)))
        print("normalize confusion_matrix")
        print(metrics.confusion_matrix(y_test, np.round(y_pred_prob), normalize='true'))
        cm1 = metrics.confusion_matrix(y_test, np.round(y_pred_prob))
        sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
        print('Sensitivity : ', sensitivity1)
        specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        print('Specificity : ', specificity1)
        return model

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
            n = 0
            total_sum = 0
            for key1 in best_parameter[key]:
                total_sum += key1 * best_parameter[key][key1]
                n += best_parameter[key][key1]
            if key == "learning_rate":
                res[key] = total_sum / n
            else:
                res[key] = int(total_sum / n)

        return res
