# Environment: pytorch

import os

import torch
from torch import nn
import torch.optim as optim

import optuna
from optuna.trial import TrialState

import functions

my_seed = 101

# IMPORT TRAINING DATA & RESERVE MOCK-TEST SET
os.chdir("..")
training_df, mytest_df = functions.get_train_test_df(fp = "train_filtered.csv", label_colname='label', my_seed = my_seed)


# TRAIN MODEL
n_features = training_df.shape[1] - 1  # number of features in feature matrix.
n_classes = len(training_df['label'].unique())  # number of unique classes.

def set_parameters(trial):
    """
    Set parameters for neural network, optimisation algorithm etc.

    :param trial: Optuna trial object

    :return: dictionary of parameters:
            - n_layers: number of layers in neural network
            - n_units_l{i}: number of units in layer i
            - dropout_l{i}: dropout probability for layer i
            - lr: learning rate
            - batch_size: batch size
            - n_epochs: number of epochs (i.e. number of passes through training data to optimise weights)
            - optimiser: optimisation algorithm to be used
    """
    trial.suggest_int("n_layers", 1, 3)

    for i in range(trial.params['n_layers']):
        trial.suggest_int(f'n_units_l{i}', 2, 20)
        trial.suggest_float(f"dropout_l{i}", 0.1, 1)

    trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    # TODO: try optimising these as well
    trial.suggest_int("batch_size", 100, 100)
    trial.suggest_int("n_epochs", 5, 5)
    trial.suggest_categorical("optimizer", ["SGD"])

    return trial.params


def define_model(my_params):
    """Defines feed-forward neural network based on set parameters

    :param my_params: dictionary of parameters (see set_parameters() for full list)

    :return: nn model
    """

    layers = []

    in_features = n_features  # number of input features for 1st layer = no. of features in feature matrix

    for i in range(my_params['n_layers']):
        # n_inputs = n_outputs of previous layer, n_outputs=no. of units in that lyr
        out_features = my_params[f'n_units_l{i}']
        layers.append(nn.Linear(in_features, out_features))

        layers.append(nn.ReLU())  # activation function

        # drop-out regularisation. (note: drop-out works by zeroing some elements of the tensor. tensor shape is unchanged)
        p = my_params[f"dropout_l{i}"]
        layers.append(nn.Dropout(p))

        in_features = out_features  # no. of inputs for next layer = no. of outputs of this layer

    layers.append(nn.Linear(in_features, n_classes))  # output layer. No. of outputs = no. of unique classes in dataset

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
study.optimize(objective, n_trials=200, timeout=600)

# Display study results and extract best trial
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("\nBest trial:")
best_trial = study.best_trial

print("  Validation Accuracy: ", best_trial.value)

print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# TRAIN FINAL MODEL USING HYPERPARAMETERS FROM BEST TRIAL
def train_final_model(my_params):
    """
    Train final model using tuned hyperparameters from best Optuna trial
    :param my_params: dictionary of parameters from Optuna trial object that had best validation accuracy

    :return: pytorch neural network model
    """

    # Instantiate model
    model = define_model(my_params)

    # Instantiate optimizer
    optimizer_name = my_params['optimizer']
    lr = my_params['lr']
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # get data. Entire training dataset is used here, including validation set
    train_dataloader = functions.df_to_dataloader(training_df, my_batchsize=my_params['batch_size'],
                                                  my_shuffle=True)

    # train model
    for epoch in range(my_params['n_epochs']):
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            # X and y are tensors. X.size() = (batch_size,n_features), y.size()=(batch_size,)
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


best_params = best_trial.params
final_model = train_final_model(best_params)

# Compute final training accuracy
train_acc = predict_and_evaluate(final_model, training_df)
print(f"  Final Training Accuracy: {train_acc}")


# EVALUATE ACCURACY ON MOCK TEST DATA
test_acc = predict_and_evaluate(final_model, mytest_df)
print(f"  Test Accuracy: {test_acc}")


# SAVE FINAL MODEL
torch.save(final_model, 'Feed Forward Neural Network/ffnn.pth')

