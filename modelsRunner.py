from Random_Forest import random_forest
from Logistic_Regression import logistic_reg
from XGBoost import xgboost
from catBoost import catboost
from LightGBM import lightGBM


class models_runner:
    def __init__(self, df):
        self.df = df
        self.modelsDic = {}

    def createNewFeature(self):
        self.df["ratio_RBC_platelets"] = self.df["Red blood Cells"] / self.df["Platelets"]
        self.df["ratio_RDW_Platelets"] = self.df["Red blood cell distribution width (RDW)"] / self.df["Platelets"]
        self.df["ratio_Hemoglobin_Platelets"] = self.df["Hemoglobin"] / self.df["Platelets"]
        self.df["ratio_Hematocrit_Hemoglobin"] = self.df["Hematocrit"] / self.df["Hemoglobin"]
        self.df["ratio_Basophils_Leukocytes"] = self.df["Basophils"] / self.df["Leukocytes"]

    def run_models(self, explore):
        if explore:
            self.createNewFeature()
        # cv1 = logistic_reg(self.df, 0.8)
        cv2 = xgboost(self.df, 0.8)
        # cv3 = random_forest(self.df, 0.8)
        # cv4 = catboost(self.df, 0.8)
        # cv5 = lightGBM(self.df, 0.8)
        # cv1.split_to_train_test()
        cv2.split_to_train_test()
        # cv3.split_to_train_test()
        # cv4.split_to_train_test()
        # cv5.split_to_train_test()

