import sklearn
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

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
        data = pd.read_excel(self.path, engine='openpyxl')
        return pd.DataFrame(data)

    def clean_empty_cols_from_csv(self):

        column_mas = self.df.isnull().mean(axis=0) < (self.thresh_hold+0.013)
        self.df =self.df.loc[:,column_mas]


    def clean_empty_rows_from_csv(self):

        nulls_number =int(len(self.df.columns) - len(self.df.columns) *0.6)
        self.df = self.df.dropna(axis=0,thresh=nulls_number)


    def drop_duplicates(self):
        self.df = self.df.drop_duplicates()
        # print("number of rows", len(self.df))
        # print("number of cols", len(self.df.columns))

    def fill_missing_values(self):
        imp = SimpleImputer(missing_values=np.NaN, strategy='median')
        trans = imp.fit_transform(self.df)
        self.df.fillna(self.df.median(), inplace=True)
    def clean_cols_non_relevant(self,name):

        self.df=self.df.drop(self.df.filter(regex=name).columns, axis=1)

    def moving_to_binary_parameter(self):
        self.df = self.df.replace(['negative'],0)
        self.df = self.df.replace(['not_detected'], 0)
        self.df = self.df.replace(['detected'], 1)
        self.df = self.df.replace(['positive'], 1)
    def clean_un_relevent_features(self):
        self.clean_cols_non_relevant("index")
        self.clean_cols_non_relevant("Bordetella pertussis")
        self.clean_cols_non_relevant("Inf A H1N1 2009")
        self.clean_cols_non_relevant("Respiratory Syncytial Virus")
        self.clean_cols_non_relevant("Strepto")
        self.clean_cols_non_relevant("Patient")
        self.clean_cols_non_relevant("Urine")
        self.clean_cols_non_relevant("fluenz")
        self.clean_cols_non_relevant("virus")

    def clean_data(self):
        self.clean_un_relevent_features()
        self.clean_empty_cols_from_csv()
        self.clean_empty_rows_from_csv()
        self.moving_to_binary_parameter()
        self.fill_missing_values()
        # print(self.df.columns)
        # col =self.df["SARS-Cov-2 exam result"]
        # count_zero=0
        # count_one =0
        # for c in col:
        #    if c== 1:
        #     count_one+=1
        #    else:
        #     count_zero+=1
        # print("countzero",count_zero)
        # print("countone", count_one)
        self.df.reset_index(inplace=True,drop=True)
        return self.df






