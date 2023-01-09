# MNIST Digit Recognizer

The objective of this project is to identify digits (0-9) from a dataset of thousands of hand-written images.

The raw data can be downloaded from the [Kaggle Digit Recognizer competition page](https://www.kaggle.com/competitions/digit-recognizer/). Due to their large file size, they are not included in this repository.

Two types of prediction models were developed:
- Feed forward neural network (FFNN)
- Convolutional neural network (CNN)

Both models were using the Pytorch library, as well as Optuna for tuning of hyperparameters.

CNN achieved the best test accuracy: 97.4%, while the FFNN achieved a significantly lower accuracy of 84.6%.

<u>Guide to file structure</u>
- Files in this main folder are common to all models.
- In each of the two folders for CNN and FFNN:
    - the model dvlpmt subfolder contains the original .py files used to build the model in PyCharm for easier debugging
    - This code has been reproduced as a Jupyter Notebook (.ipynb file) for easier viewing on GitHub
    - .pth files are saved, trained models
    - .csv files are predictions on the test set