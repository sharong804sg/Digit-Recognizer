# Environment: pytorch

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import optuna
from optuna.trial import TrialState

import functions

my_seed = 101

# IMPORT TRAINING DATA & RESERVE MOCK-TEST SET
os.chdir("..")
training_df, mytest_df = functions.get_train_test_df(fp = "train.csv", label_colname='label', my_seed = my_seed)

# PLOT A FEW DIGITS (to check code in MyDataset used to reassemble images)
check_dataset = functions.MyDataset(training_df, 'label')

check_dataloader = DataLoader(check_dataset, batch_size=5)

check_iter = iter(check_dataloader)
images, labels = next(check_iter) # returns 5 random training images and their labels

fig, ax = plt.subplots(figsize=(15, 2.5), ncols=5)
ax = ax.flatten()
for i in range(len(images)):
    ax[i].imshow(np.transpose(images[i], (1, 2, 0)))
    ax[i].axis('off')
    ax[i].set_title(f"Label: {labels[i].item()}")

plt.show()


# TRAIN MODEL
def set_parameters(trial):
    """
    Set parameters for neural network, optimisation algorithm etc.
    :param trial: Optuna trial object
    :return: dictionary of parameters:
            - n_conv_layers: number of convolution layers in neural network
            - out_ch_conv{i}: number of output channels in convolution layer i
            - kernel_conv{i}_even: kernel width in convolution layer i - even option
            - kernel_conv{i}_odd:                                      - odd option

            - n_linear_layers: number of linear layers in neural network
            - n_units_lin{i}: number of units in linear layer i
            - dropout_lin{i}: dropout probability for linear layer i

            - lr: learning rate
            - batch_size: batch size
            - n_epochs = number of epochs (i.e. number of passes through training data during optimisation)
    """
    trial.suggest_int("n_conv_layers", 1, 2)

    for i in range(trial.params['n_conv_layers']):
        trial.suggest_int(f'out_ch_conv{i}', 1, 20)
        trial.suggest_categorical(f'kernel_conv{i}_even', [2, 4, 6])
        trial.suggest_categorical(f'kernel_conv{i}_odd', [3, 5, 7])

    trial.suggest_int("n_linear_layers", 1, 3)

    for i in range(trial.params['n_linear_layers']):
        trial.suggest_int(f'n_units_lin{i}', 1, 200)
        trial.suggest_float(f"dropout_lin{i}", 0.1, 0.9)

    trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # TODO: try optimising these as well
    trial.suggest_int("batch_size", 10, 10)
    trial.suggest_int("n_epochs", 5, 5)
    trial.suggest_categorical("optimizer",["SGD"])

    my_params = trial.params

    return my_params


def define_model(my_params):
    """Defines convolutional neural network based on set parameters
    :param my_params: dictionary of parameters (see set_parameters() for full list)
    """

    layers = []

    # Define Convolution Layers
    in_ch = 1  # number of input channels = no. of channels in feature matrix = 1
    img_width = 28 # number of px along length & width of feature matrix
    for i in range(my_params['n_conv_layers']):
        # convolution layer
        out_ch = my_params[f'out_ch_conv{i}']  # number of output channels for this layer
        # for even image width use odd kernel width so that resulting img width is divisible by 2 during pooling
        if (img_width % 2) == 0:
            kernel_size = my_params[f'kernel_conv{i}_odd']
        else:
            kernel_size = my_params[f'kernel_conv{i}_even']
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size))

        layers.append(nn.ReLU())  # activation function
        layers.append(nn.MaxPool2d(2, 2))  # pooling layer

        in_ch = out_ch  # no. of input channels for next layer = no. of output channels from this layer
        img_width = int((img_width-(kernel_size-1))/2)

    layers.append(nn.Flatten(start_dim=1))  # flatten all dimensions except batch

    # Define Linear Layers
    in_features = in_ch * img_width * img_width
    for i in range(my_params['n_linear_layers']):
        # linear layer
        out_features = my_params[f'n_units_lin{i}']
        layers.append(nn.Linear(in_features, out_features))

        layers.append(nn.ReLU())  # activation function

        #drop-out regularisation
        p = my_params[f"dropout_lin{i}"]
        layers.append(nn.Dropout(p))

        in_features = out_features  # no. of inputs for next layer = no. of outputs of this layer

    layers.append(nn.Linear(in_features, 10))  # output layer

    return nn.Sequential(*layers)

def objective(trial):
    """
    Objective for Optuna to optimise
    :param trial: Optuna trial object
    :return: accuracy - fraction of correctly labelled validation points. This is what Optuna seeks to maximise
    """

    #set parameters
    my_params = set_parameters(trial)

    # Instantiate model
    model = define_model(my_params)

    # Instantiate optimizer
    optimizer_name = my_params['optimizer']
    lr = my_params['lr']
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # get data
    train_dataloader, val_dataloader = functions.get_train_val_dataloader(training_df,
                                                                          my_batchsize=my_params['batch_size'],
                                                                          label_colname='label')

    # train model
    for epoch in range(my_params['n_epochs']):

        #train
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            # X and y are tensors. X.size() = (batch_size,n_features), y.size()=(batch_size,)
            # set datatype for compatibility with nn.
            X = X.float()
            y = y.long()

            # calculate model output and resulting loss
            model_output = model(X)  # tensor. size=(batch_size x n_classes)
            loss_fn = nn.CrossEntropyLoss() # instantiate loss function
            loss = loss_fn(model_output, y)

            # Backpropagation to update model weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validate. We do this at each epoch to facilitate pruning:
        # i.e. early termination of trials which are clearly not going to be optimum
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch, (X, y) in enumerate(val_dataloader):
                X = X.float()
                y = y.long()

                # calculate model output and total number of correct predictions for this batch
                model_output = model(X)
                pred = torch.argmax(model_output, dim=1)  # prediction = class with highest output value
                correct += functions.count_correct(pred, y)

        accuracy = correct / len(val_dataloader.dataset)

        # report accuracy to allow Optuna to decide whether to prune this trial
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy  # return final validation accuracy after all epochs (unless pruned)

# instantiate optuna study
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
# Optimise hyperparameters will try {n_trials} param combinations or till {timeout} seconds is hit
study.optimize(objective, n_trials=100)  # , timeout=600)

#display study results
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
best_trial = study.best_trial

print("  Validation Accuracy: ", best_trial.value)

print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")


# TRAIN FINAL MODEL USING TUNED HYPER-PARAMETERS
def train_final_model(my_params):
    """
    Train final model using tuned hyperparameters from best Optuna trial
    :param my_params: dictionary of parameters from Optuna trial object that had best validation accuracy

    :return: model
    """

    # Instantiate model
    model = define_model(my_params)

    # Instantiate optimizer
    optimizer_name = my_params['optimizer']
    lr = my_params['lr']
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # get data
    train_dataloader = functions.df_to_dataloader(training_df, my_batchsize=my_params['batch_size'],
                                                  my_shuffle=True)

    # train model
    for epoch in range(my_params['n_epochs']):
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            # set datatype for compatibility with nn.
            X = X.float()
            y = y.long()

            # calculate model output and resulting loss
            model_output = model(X)  # tensor. size=(batch_size x n_classes)
            loss_fn = nn.CrossEntropyLoss()  # instantiate loss function
            loss = loss_fn(model_output, y)

            # Backpropagation to update model weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


best_params = best_trial.params
final_model = train_final_model(best_params)


# EVALUATE FINAL TRAINING ACCURACY
def predict_and_evaluate(model, df):
    """
    Function to run trained and tuned model on provided dataframe to obtain predictions and evaluate
    accuracy

    :param model: trained model
    :param df: dataframe including features and target/label

    :return: accuracy
    """
    my_dataloader = functions.df_to_dataloader(df, my_batchsize=10, my_shuffle=False)

    model.eval()
    correct = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(my_dataloader):
            X = X.float()
            y = y.long()

            # calculate model output and total number of correct predictions for this batch
            model_output = model(X)
            pred = torch.argmax(model_output, dim=1)  # prediction = class with highest output value
            correct += functions.count_correct(pred, y)

    accuracy = correct / len(my_dataloader.dataset)

    return accuracy


train_acc = predict_and_evaluate(final_model, training_df)
print(f"  Final Training Accuracy: {train_acc}")


# EVALUATE ACCURACY ON MOCK TEST DATA
test_acc = predict_and_evaluate(final_model, mytest_df)
print(f"  Test Accuracy: {test_acc}")


# SAVE FINAL MODEL
torch.save(final_model, 'Convolutional Neural Network/cnn_1.pth')