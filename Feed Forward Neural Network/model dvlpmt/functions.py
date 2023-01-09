# Environment: pytorch

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def get_train_test_df(fp, label_colname, my_seed=None):
    """
    Function to import raw data, carry out pre-processing, and split into training and test datasets.
    Test data will be reserved for final evaluation of model performance (i.e. not for hyperparameter tuning)

    :param fp: filepath
    :param label_colname: name of column containing labels
    :param my_seed: integer to be used to fix random state for train_test_split

    :return: tuple of dataframes - training_df, test_df
    """

    # import data
    df = pd.read_csv(fp)

    # Standard scaling of features #TODO: try with and without
    scaler = StandardScaler()
    df[df.drop(columns=label_colname).columns] = scaler.fit_transform(df[df.drop(columns=label_colname).columns])

    # separate into training & test datasets.
    # Stratification is used to ensure training and test sets have representative proportions of all classes
    training_df, test_df = train_test_split(df, test_size=0.3, random_state=my_seed, stratify=df[label_colname])

    return training_df, test_df


class MyDataset(Dataset): # inherits properties of pytorch Dataset class
    def __init__(self, dataframe, label_colname=None, blind_test=False):
        """
            Class initialisation
            :param dataframe: pandas dataframe including features and labels
            :param label_colname: name of column containing labels
            """
        self.blind_test = blind_test

        if blind_test:  # for blind test (i.e. no label, self.labels does not exist)
            self.features = dataframe.to_numpy()
        else:
            self.features = dataframe.drop(columns=[label_colname]).to_numpy()
            self.labels = dataframe[label_colname].to_numpy()


    def __len__(self):
        """
        :return: length of dataset
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Fetches features and label(s) at requested index
        :param idx: requested index
        :return: tuple of numpy arrays - batch_features, batch_labels. For blind test, return only batch_features
        """
        batch_features = self.features[idx,:]
        if self.blind_test:
            return batch_features
        else:
            batch_labels = self.labels[idx]
            return batch_features, batch_labels


def get_train_val_dataloader(training_df, my_batchsize, label_colname, my_seed = None):
    """
    Function to split training data into training and validation subsets and format as dataloaders
    Model performance on validation set will be used for hyperparameter tuning.

    :param training_df: dataframe with full set of training data
    :param my_batchsize: batch size for pytorch DataLoader
    :param label_colname: name of column containing labels
    :param my_seed: optional integer to fix train test split random state

    :return: tuple of pytorch DataLoaders - train_dataloader, val_dataloader
    """

    # separate into training & validation datasets
    train_data, val_data = train_test_split(training_df, test_size = 0.2, random_state = my_seed, stratify=training_df[label_colname])

    #format as pytorch dataloader
    train, val = MyDataset(train_data, label_colname), MyDataset(val_data, label_colname)
    train_dataloader = DataLoader(train, batch_size=my_batchsize, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=my_batchsize)

    return train_dataloader, val_dataloader


def count_correct(predictions, y):
    """
    Counts number of correct predictions in a batch

    :param predictions: 1D tensor with predictions
    :param y: 1D tensor with true classes

    :return: number of correct predictions (pred==y)
    """
    predictions = predictions.numpy()
    y = y.numpy()

    n_correct = (predictions == y).sum()

    return n_correct


def df_to_dataloader(df, my_batchsize, my_shuffle, blind_test = False):
    """
    Function to format dataframe as dataloader
    :param df: dataframe
    :param blind_test: true if df has no labels
    :param my_batchsize: batch size for dataloader
    :return: dataloader
    """
    data = MyDataset(df, 'label', blind_test)
    my_dataloader = DataLoader(data, batch_size=my_batchsize, shuffle=my_shuffle)

    return my_dataloader