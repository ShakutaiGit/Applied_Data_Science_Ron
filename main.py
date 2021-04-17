# This is a sample Python script.
from sklearn.model_selection import train_test_split
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from Pre_Processing import pre_processing
from Cross_valid import cross_valid_and_hyper
def split_data_to_train_test(df,splited_precent):

    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = 'C:\\Users\\ronsh\\PycharmProjects\\pythonProject2\\Data\dataset\\Bloot_Test_dataset\\dataset.xlsx'
    pp = pre_processing(path,0.95)
    pp.read_data_csv()
    clean_df = pp.clean_data()
    cv = cross_valid_and_hyper(clean_df,0.8)
    cv.split_to_train_test()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
