import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import sys
sys.path.append("/home/rishi/Codes/covid19sim")
from src.dataset.covid19india_national_loader import Covid19IndiaNationalLoader

dataloader = Covid19IndiaNationalLoader()
data = dataloader.load()

data = data[['TotalConfirmed']].to_numpy() 
print(data.shape)
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# fig_size = plt.rcParams["figure.figsize"]
# fig_size[0] = 15
# fig_size[1] = 5
# plt.rcParams["figure.figsize"] = fig_size

# plt.title('days vs cases')
# plt.ylabel('cases')
# plt.xlabel('days')
# plt.grid(True)
# plt.autoscale(axis='x',tight=True)
# plt.plot(data)
# plt.show()

test_data_size = int(0.1*data.shape[0])

train_data = data[:-test_data_size]
test_data = data[-test_data_size:]

print("train data size", train_data.shape)
print("test data size", test_data.shape)

## normalise here 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))

train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


train_window = 10
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 150

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

fut_pred = test_data.shape[0]

test_inputs = train_data_normalized[-train_window:].tolist()
print(test_inputs)


model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())



actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))

x = np.arange(train_data.shape[0], data.shape[0], 1)

plt.title('days vs cases')
plt.ylabel('cases')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(data)
plt.plot(x,actual_predictions)
plt.show()


plt.title('days vs cases')
plt.ylabel('cases')
plt.grid(True)
plt.autoscale(axis='x', tight=True)

plt.plot(data[-train_window:])
plt.plot(actual_predictions)
plt.show()