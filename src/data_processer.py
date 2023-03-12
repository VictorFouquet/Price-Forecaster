import numpy as np
import pandas

class DataProcesser:
    def __init__(self, csv_path, total_time, period, target_label='', exclude_labels=[]):
        """
        Creates a data processer.
            Args:
                csv_path (str): path to the dataset csv
                total_time (int): number of rows to use
                period (int): number of rows per input
                target_label (str): label of the target column
                exclude_labels (list(str)): columns to drop 
        """
        self.df = pandas.read_csv(csv_path)  # Raw csv dataframe
        self.df_clean = None                 # Clean csv dataframe
        self.total_time = total_time         # Total time to study (number of rows)
        self.period = period                 # Period for each input (previous days included in one input)
        self.target_label = target_label     # Label of the target column
        self.exlude_labels = exclude_labels  # Labels of the columns to drop
        self.columns_len = 0                 # Number of columns after drop
        self.max_Y = 0                       # Max value in target column (used to scale back target)

    def preprocess(self):
        """
        Pre-processes data, droping columns, interpolating missing data, slicing and scaling.
        """
        # Drops unused columns
        self.df_clean = self.df.drop(self.exlude_labels, axis=1)
        self.columns_len = len(self.df_clean.columns)
        # Interpolates missing values
        self.df_clean = self.df_clean.interpolate(method='linear', limit_direction='forward')
        # Slices dataset's rows
        start = len(self.df_clean) - (self.total_time + self.period)
        self.df_clean = self.df_clean.loc[start:, :]
        # Scales dataset to range between 0 and 1
        self.max_Y = self.df_clean[self.target_label].max()
        self.df_clean = self.df_clean / self.df_clean.max()

    def build_sets(self):
        """
        Builds training and testing datasets
        """
        # Stores all targets and inputs
        Y = np.zeros(self.total_time)
        X = np.zeros((self.total_time, self.period * len(self.df_clean.columns)))
        for i in range(self.period, self.total_time + self.period):
            # Stores target
            Y[i - self.period] = self.df_clean[self.target_label].values[i]
            # Packs period rows into one input and store it
            X[i - self.period] = np.array(self.df_clean.values[i - self.period : i]).flatten()
        # Computes pivot to use 80% of the data for training, 20% for testing
        pivot = int(np.ceil(self.total_time * 0.8))
        # Creates training data
        X_train = X[:pivot]
        Y_train = Y[:pivot]
        # Creates testing data
        X_test  = X[pivot:]
        Y_test  = Y[pivot:]
        return X_train, Y_train, X_test, Y_test
