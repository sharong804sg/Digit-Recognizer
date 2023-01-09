import os
import pandas as pd
import numpy as np
import torch
import functions


# import test set
os.chdir("..")
test_df = pd.read_csv("test_filtered.csv")

# convert to dataloader
test_dataloader = functions.df_to_dataloader(test_df, my_batchsize=100, my_shuffle=False, blind_test=True)

# import trained model
model = torch.load('Feed Forward Neural Network/ffnn.pth')

# make predictions
model.eval()
with torch.no_grad():
    for batch, X in enumerate(test_dataloader):
        X = X.float()
        model_output = model(X)
        pred = torch.argmax(model_output, dim=1).numpy()  # prediction = class with highest output value
        if batch==0:
            predictions = pred
        else:
            predictions = np.append(predictions, pred)

image_id = np.arange(1,len(predictions)+1)
predictions_df = pd.DataFrame.from_dict(data={'ImageId': image_id, 'Label': predictions})

predictions_df.to_csv('Feed Forward Neural Network/mysubmission_ffnn.csv', index=False)

# test accuracy (from Kaggle: 0.823)