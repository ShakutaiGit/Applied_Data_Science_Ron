from Random_Forest import random_forest
from Logistic_Regression import logistic_reg
from XGBoost import xgboost
from catBoost import catboost
from LightGBM import lightGBM


class models_runner:
    def __init__(self, df):
        self.df = df
        self.modelsDic = {}
        self.modelsDic["log_reg"]=None
        self.modelsDic["rf"] = None
        self.modelsDic["xg"] = None
        self.modelsDic["cat"] = None
        self.modelsDic["light"] = None

    def createNewFeature(self):
        self.df["ratio_RBC_platelets"] = self.df["Red blood Cells"] / self.df["Platelets"]
        self.df["ratio_RDW_Platelets"] = self.df["Red blood cell distribution width (RDW)"] / self.df["Platelets"]
        self.df["ratio_Hemoglobin_Platelets"] = self.df["Hemoglobin"] / self.df["Platelets"]
        self.df["ratio_Hematocrit_Hemoglobin"] = self.df["Hematocrit"] / self.df["Hemoglobin"]
        self.df["ratio_Basophils_Leukocytes"] = self.df["Basophils"] / self.df["Leukocytes"]

    def run_models(self, explore):
        if explore:
            self.createNewFeature()
        self.modelsDic["log_reg"] = logistic_reg(self.df, 0.8).split_to_train_test()
        self.modelsDic["rf"] = random_forest(self.df, 0.8).split_to_train_test()
        self.modelsDic["xg"] = xgboost(self.df, 0.8).split_to_train_test()
        self.modelsDic["cat"] = catboost(self.df, 0.8).split_to_train_test()
        self.modelsDic["light"] = lightGBM(self.df, 0.8).split_to_train_test()
        return self.modelsDic

