import tqdm as tqdm
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
class cross_valid_and_hyper:
    def __init__(self,df,prec):
        self.df = df
        self.split_prec=prec

    def split_to_train_test(self,model):
        print(self.df)
        y = self.df["SARS-Cov-2 exam result"]
        X = self.df.drop(["SARS-Cov-2 exam result"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7,shuffle=True)
        X_train.reset_index(inplace=True,drop=True)
        cv_outer = StratifiedKFold(n_splits=5,random_state=7,shuffle=True)
        for train_idx, val_idx in tqdm.tqdm(cv_outer.split(X_train, y_train)):
            train_data, val_data = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train.reset_index(inplace=True,drop=True)
            train_target, val_target = y_train[train_idx], y_train[val_idx]
            cv_inner = StratifiedKFold(n_splits=5, random_state=7)

