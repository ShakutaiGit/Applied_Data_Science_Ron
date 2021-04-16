import sklearn
import pandas as pd

class pre_processing:
    def __init__(self, path,thresh_hold):
        self.thresh_hold = thresh_hold
        self.path = path
        self.df = self.read_data_csv()
#Pre Processing from article
#To select the most representative parameters in the dataset we first
# define a threshold of 95% for removing features with several missing
# values greater than it. Non-blood features were also discarded, such as
# urine tests and other contagious infectious diseases. These diseases
# include respiratory infections, such as influenza A and B; parainfluenza
# 1, 2, 3 and 4; enterovirus infections and others. We remove these features
# since the dependence of the diagnosis on a variety of other infectious
# diseases for COVID-19 prediction is not a practical situation in the
#emergency context. Furthermore, a false negative result of one of these
# diseases would generate a spread of the error.
# However, the diagnostic results for the others infectious diseases
# could be used to train a multiple output classifier, which may assist the
# health professional in the process of diagnosing simultaneous diseases.
# But this is not the focus of this work.
# The set of final features were detailed in Table 1. After the cleaning
# process, we found a total of 608 observations, being 84 positive and 524
# negative COVID-19 confirmed cases through RT-PCR being, thus, an
# imbalanced data problem. The distribution for each class is approximately
# 1:6 ratio. Since many null values remained, it was necessary an
# imputation technique to deal with. The “Iterative Imputer” technique
# from Scikit-learn package [44] showed the best performance in experimental
# tests compared with mean or median.
#C:\\Users\\ronsh\\PycharmProjects\\pythonProject2\\Data\dataset\\Bloot_Test_dataset\\dataset.xlsx
    def read_data_csv(self):
        data=pd.read_excel(self.path)
        return pd.DataFrame(data)

    def clean_empty_cols_from_csv(self):
        print("number of rows", len(self.df))
        print("number of cols", len(self.df.columns))
        column_mas = self.df.isnull().mean(axis=0) < self.thresh_hold
        self.df =self.df.loc[:,column_mas]
        print("number of rows", len(self.df))
        print("number of cols", len(self.df.columns))

    def clean_empty_rows_from_csv(self):
        print("number of rows", len(self.df))
        print("number of cols", len(self.df.columns))
        nulls_number = len(self.df.columns) * 0.5
        self.df = self.df.dropna(axis=0,thresh=nulls_number)
        print("number of rows", len(self.df))
        print("number of cols", len(self.df.columns))

    def drop_duplicates(self):
        print("number of rows", len(self.df))
        print("number of cols", len(self.df.columns))
        self.df = self.df.drop_duplicates()
        print("number of rows", len(self.df))
        print("number of cols", len(self.df.columns))

    def clean_nulls_attributes_according_to_thresh_hold(self):
        pass

    def clean_data(self):
        self.drop_duplicates()
        self.clean_empty_cols_from_csv()
        self.clean_empty_rows_from_csv()
