# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from Pre_Processing import pre_processing

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.

    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = 'C:\\Users\\ronsh\\PycharmProjects\\pythonProject2\\Data\dataset\\Bloot_Test_dataset\\dataset.xlsx'
    pp = pre_processing(path,0.95)
    pp.read_data_csv()
    pp.clean_data()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
