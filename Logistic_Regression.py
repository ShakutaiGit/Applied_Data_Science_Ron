from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split
import tqdm as tqdm
from sklearn.linear_model import LogisticRegression
import numpy as np
class logistic_reg:
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
        best_parameters = {'C':{},'penalty':{},'class_weight':{},'solver':{}}
        for train_idx, val_idx in tqdm.tqdm(cv_outer.split(X_train, y_train)):
            train_data, val_data = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train.reset_index(inplace=True, drop=True)
            train_target, val_target = y_train[train_idx], y_train[val_idx]
            cv_inner = StratifiedKFold(n_splits=5, random_state=7,shuffle=True)
            model = LogisticRegression(max_iter=200)
            params = {'penalty': ['l1', 'l2'], 'solver': ['liblinear'], 'C': [10**x for x in range(-3,5)],'class_weight':['balanced',None]}
            # print(model.get_params().keys())
            gd_search = GridSearchCV(model, params, scoring='roc_auc', n_jobs=-1, cv=cv_inner).fit(train_data,train_target)
            best_model = gd_search.best_estimator_
            classifier = best_model.fit(train_data, train_target)
            y_pred_prob = classifier.predict_proba(val_data)[:, 1]
            auc = metrics.f1_score(val_target, y_pred_prob)
            print("Val Acc:", auc, "Best GS Acc:", gd_search.best_score_, "Best Params:", gd_search.best_params_)
            self.update_the_best_parameter(best_parameters,gd_search.best_params_)


        print(best_parameters)
        mean_best_parameters = self.get_top_values_dict(best_parameters)
        print(X_train)
        model = LogisticRegression(multi_class='auto',C=mean_best_parameters['C'], class_weight=mean_best_parameters['class_weight'], penalty=mean_best_parameters['penalty'],solver=mean_best_parameters['solver']).fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:,1]
        print("AUC", metrics.roc_auc_score(y_test, y_pred_prob))
        print(metrics.confusion_matrix(y_test, y_pred_prob))

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
