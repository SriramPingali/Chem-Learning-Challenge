import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_data = pd.read_csv("CLC_test.csv")
labels_test_pd = test_data.pop("CO_level")
include_columns = ["CO_GT", "PT08_S1_CO", "PT08_S2_NMHC"]
data_test_pd = test_data[include_columns]
X_test = data_test_pd.to_numpy()
Y_test = labels_test_pd.to_numpy()
enc = OneHotEncoder()
Y_OH_test = enc.fit_transform(np.expand_dims(Y_test, 1)).toarray()
X_test, Y_OH_test = map(torch.tensor, (X_test, Y_OH_test))
X_test = X_test.float()
X_test = X_test.to(device)
Y_OH_test = Y_OH_test.to(device)

class FF_Network(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            nn.Linear(3, 55),
            nn.BatchNorm1d(55),
            nn.ReLU(),
            nn.Linear(55,40),
            nn.BatchNorm1d(40),
            nn.ReLU(), 
            nn.Linear(40, 5),
        )
# 9, 256, 40, 5
# 9, 64, 40, 5
# 3, 50, 40, 5
           
    def forward(self, X):
        return(self.net(X))

def accuracy(Y_hat, Y):
    accuracy = accuracy_score(np.argmax(Y_hat.cpu().detach().numpy(), 1), np.argmax(Y.cpu().detach().numpy(), 1))
    f1_scr = f1_score(np.argmax(Y_hat.cpu().detach().numpy(), 1), np.argmax(Y.cpu().detach().numpy(), 1), average='macro')
    kappa = cohen_kappa_score(np.argmax(Y_hat.cpu().detach().numpy(), 1), np.argmax(Y.cpu().detach().numpy(), 1))
    return(accuracy, f1_scr, kappa)

model = FF_Network()
model.load_state_dict(torch.load('./weights'))
model.to(device)
model.eval()

Y_test_pred = model.forward(X_test)
acc, f1, kappa = accuracy(Y_test_pred, Y_OH_test)

print("Test set Accuracy: ", acc, "\nF1 score : ", f1, "\nCohen's Kappa: ", kappa)

class_label = ["High", "Low", "Moderate", "Very High", "Very Low"]

import csv

with open('submission_trials.csv', 'w', newline='') as file:
    with open('CLC_test.csv', 'r') as inp:
        writer = csv.writer(file)
        reader = csv.reader(inp)
        heading = next(reader)
        heading.append("Our Prediction")
        writer.writerow(['Date', 'Time', 'CO_GT', 'PT08_S1_CO', 'NMHC_GT', 'C6H6_GT', 'PT08_S2_NMHC', 'Nox_GT', 'PT08_S3_Nox', 'NO2_GT', 'PT08_S4_NO2', 'PT08_S5_O3', 'T', 'RH', 'AH', 'CO_level', 'Our prediction'])
        for i, row in enumerate(reader):
            row.append(class_label[np.argmax(Y_test_pred.to('cpu').detach().numpy(), 1)[i]])
            writer.writerow(row)
print("""\nThe predictions are now stored in submission_trials.csv under "Our prediction" column!""")