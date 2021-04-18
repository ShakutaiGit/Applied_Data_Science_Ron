# This is a sample Python script.
from sklearn.model_selection import train_test_split
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from Pre_Processing import pre_processing
from Random_Forest import random_forest
from Logistic_Regression import logistic_reg
from XGBoost import xgboost



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = "C:\\Users\\Lenovo\\Desktop\\לימודים\\יישומית\\dataset\\Bloot Test dataset\\dataset.xlsx"
    pp = pre_processing(path, 0.95)
    pp.read_data_csv()
    clean_df = pp.clean_data()
    # cv = logistic_reg(clean_df, 0.8)
    cv = xgboost(clean_df, 0.8)
    # cv = random_forest(clean_df, 0.8)
    cv.split_to_train_test()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
