# MNIST Digit Recognizer

The objective of this project is to identify digits (0-9) from a dataset of thousands of hand-written images.

The raw data (test.csv, train.csv) are from the [Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer/), and are made available in this repository under the  [Creative Commons Attribution-Share Alike 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/).

Two types of prediction models were developed:
- Feed forward neural network (FFNN)
- Convolutional neural network (CNN)

Both models were using the Pytorch library, as well as Optuna for tuning of hyperparameters.

CNN achieved the best test accuracy: 97.4%, while the FFNN achieved a significantly lower accuracy of 84.6%.

<u>Guide to file structure</u>
- train.csv and test.csv are the original files from Kaggle
- train_filtered.csv and test_filtered.csv are processed files (output from data cleaning & exploration)
- In each of the two folders for CNN and FFNN:
    - the model dvlpmt subfolder contains the original .py files used to build the model in PyCharm for easier debugging
    - This code has been reproduced as a Jupyter Notebook (.ipynb file) for easier viewing on GitHub
    - .pth files are saved, trained models
    - .csv files are predictions on the test set