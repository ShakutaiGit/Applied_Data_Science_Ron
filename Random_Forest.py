from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
import tqdm as tqdm
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import shap
import matplotlib.pyplot as plt


class random_forest:
    def __init__(self, df, prec):
        self.df = df
        self.split_prec = prec

    def split_to_train_test(self):
        y = self.df["SARS-Cov-2 exam result"]
        X = self.df.drop("SARS-Cov-2 exam result", axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, shuffle=True)
        X_train.reset_index(inplace=True, drop=True)
        cv_outer = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
        max_score = 0
        best_parameters = {'max_depth': {}, 'n_estimators': {}}
        for train_idx, val_idx in tqdm.tqdm(cv_outer.split(X_train, y_train)):
            train_data, val_data = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train.reset_index(inplace=True, drop=True)
            train_target, val_target = y_train[train_idx], y_train[val_idx]
            rf = RandomForestRegressor(n_estimators=20, random_state=0)
            cv_inner = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
            params = {
                'max_depth': [2, 4, 8, 16, 32, 64],
                'n_estimators': [10, 20, 30, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
            }
            # print(model.get_params().keys())
            gd_search = GridSearchCV(rf, params, scoring='roc_auc', n_jobs=-1, cv=cv_inner).fit(train_data, train_target)
            best_model = gd_search.best_estimator_
            classifier = best_model.fit(train_data, train_target)
            y_pred_prob = classifier.predict(val_data)
            # auc = metrics.roc_auc_score(val_target, y_pred_prob)
            # print("Val Acc:", auc, "Best GS Acc:", gd_search.best_score_, "Best Params:", gd_search.best_params_)
            self.update_the_best_parameter(best_parameters, gd_search.best_params_)

        mean_best_parameters = self.get_top_values_dict(best_parameters)
        model = RandomForestRegressor(max_depth=mean_best_parameters['max_depth'], n_estimators=mean_best_parameters['n_estimators']).fit(X_train, y_train)
        y_pred_prob = model.predict(X_test)
        print("random forest")
        print("rocAUC", metrics.roc_auc_score(y_test, y_pred_prob))
        print("f1", metrics.f1_score(y_test, np.round(y_pred_prob)))
        print("accuracy_score", metrics.accuracy_score(y_test, np.round(y_pred_prob)))
        # print(metrics.classification_report(y_test, np.round(y_pred_prob)))
        print("normalize confusion_matrix")
        print(metrics.confusion_matrix(y_test, np.round(y_pred_prob), normalize='true'))
        cm1 = metrics.confusion_matrix(y_test, np.round(y_pred_prob))
        sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
        print('Sensitivity : ', sensitivity1)
        specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        print('Specificity : ', specificity1)
        self.plotter_function(model,X_train,X_test)

    def plotter_function(self,model,X_train,X_test):
        #ploting Variable Importance Plot — Global Interpretability
        shap_values = shap.TreeExplainer(model).shap_values(X_train)
        shap.summary_plot(shap_values, X_train, plot_type="bar")
        f = plt.figure()
        shap_values = shap.TreeExplainer(model).shap_values(X_test)
        shap.summary_plot(shap_values, X_test)



    def update_the_best_parameter(self,best_parameter,iter):
        for key in iter.keys():
            value =iter[key]
            if value in best_parameter[key]:
                best_parameter[key][value]+=1
            else:
                best_parameter[key][value]=1

    def get_top_values_dict(self, best_parameter):
        res = {}
        for key in best_parameter.keys():
            n = 0
            total_sum = 0
            for key1 in best_parameter[key]:
                total_sum += key1 * best_parameter[key][key1]
                n += best_parameter[key][key1]
            res[key] = int(total_sum / n)

        return res

