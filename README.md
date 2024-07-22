# PM2.5 and Relative Humidity Forecasting
<p align="center">
<img src="https://raw.githubusercontent.com/JNDEV03/Pm2.5-Chiang-Mai-Predic/main/Screenshot%202024-07-22%20001446.png"/> </a> 
</p>

## Overview

This project aims to forecast PM2.5 levels and relative humidity (RH) using historical data from a specified station. The data is processed and used to train a Long Short-Term Memory (LSTM) neural network to predict future values.

## Steps

### Step 1: Fetch Historical Data

Historical data for PM2.5 and RH is fetched from an online source using an API.

### Step 2: Preprocess Data

The data is preprocessed to ensure it is in the correct format for training the model:
- Convert datetime strings to `datetime` objects.
- Set the datetime column as the index.
- Convert relevant columns to float.
- Fill missing values using forward fill.

### Step 3: Split the Data

The dataset is split into training and testing sets:
- Training set: All data except the last 7 days.
- Testing set: Last 14 days of data.

### Step 4: Define LSTM Model

A PyTorch LSTM model is defined with the following architecture:
- Input size: 2 (PM2.5 and RH)
- Hidden size: 100
- Output size: 2
- Number of layers: 2
- Dropout probability: 0.2

### Step 5: Train the Model

The model is trained on the training dataset:
- Batch size: 16
- Number of epochs: 500
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam

Early stopping is implemented to prevent overfitting.

### Step 6: Evaluate the Model

The model is evaluated on the testing dataset to calculate the Mean Squared Error (MSE).

### Step 7: Predict Future Values

The trained model is used to predict future PM2.5 and RH values for the next four weeks.

### Plotting the Results

The results are plotted with danger zones to highlight risk levels for PM2.5 and RH.

## Code

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 1: Fetch Historical Data
url = "http://air4thai.com/forweb/getHistoryData.php?stationID=35t&param=PM25,RH&type=hr&sdate=2024-06-21&edate=2024-07-21&stime=00&etime=23"
response = requests.get(url)
data = response.json()

# Step 2: Preprocess Data
df = pd.DataFrame(data['stations'][0]['data'])
df.rename(columns={'DATETIMEDATA': 'datetime'}, inplace=True)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
df = df[['PM25', 'RH']].astype(float)
df.fillna(method='ffill', inplace=True)

# Step 3: Split the Data
train_df = df[:-24*7]
test_df = df[-24*14:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (
            torch.FloatTensor(self.data[index:index + self.seq_length]),
            torch.FloatTensor(self.data[index + self.seq_length])
        )

seq_length = 24
train_dataset = TimeSeriesDataset(train_scaled, seq_length)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Step 4: Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 2
hidden_size = 100
output_size = 2
num_layers = 2
dropout_prob = 0.2

model = LSTMModel(input_size, hidden_size, output_size, num_layers, dropout_prob).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train the Model
num_epochs = 500
best_loss = float('inf')
patience = 10
trigger_times = 0

for epoch in range(num_epochs):
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Early stopping
    if loss.item() < best_loss:
        best_loss = loss.item()
        trigger_times = 0
        best_model = model.state_dict()

# Load the best model
model.load_state_dict(best_model)

# Step 6: Evaluate the Model
model.eval()
test_inputs = torch.FloatTensor(test_scaled[:seq_length]).unsqueeze(0).to(device)
with torch.no_grad():
    test_preds = []
    for i in range(len(test_scaled) - seq_length):
        pred = model(test_inputs)
        test_preds.append(pred.squeeze().cpu().numpy())
        test_inputs = torch.cat((test_inputs[:, 1:, :], pred.unsqueeze(0)), dim=1)
    test_preds = scaler.inverse_transform(test_preds)

test_mse = mean_squared_error(test_df.values[seq_length:], test_preds)
print(f'Test MSE: {test_mse:.4f}')

# Step 7: Predict Future Values
with torch.no_grad():
    future_inputs = torch.FloatTensor(test_scaled[-seq_length:]).unsqueeze(0).to(device)
    future_preds = []
    for _ in range(24 * 7 * 4):  # Predicting for four weeks
        pred = model(future_inputs)
        future_preds.append(pred.squeeze().cpu().numpy())
        future_inputs = torch.cat((future_inputs[:, 1:, :], pred.unsqueeze(0)), dim=1)
    future_preds = scaler.inverse_transform(future_preds)

# Convert predictions to DataFrame
future_index = pd.date_range(start=test_df.index[-1], periods=len(future_preds)+1, freq='H')[1:]
future_df = pd.DataFrame(future_preds, index=future_index, columns=['PM25', 'RH'])

# Plotting the results with Danger Zones
plt.figure(figsize=(14, 12))

plt.subplot(3, 1, 1)
plt.plot(df['PM25'], label='Historical PM2.5')
plt.plot(test_df.index[seq_length:], test_preds[:, 0], label='Test Predicted PM2.5')
plt.axhline(y=35, color='r', linestyle='--', label='Moderate Risk (PM2.5)')
plt.axhline(y=55.4, color='y', linestyle='--', label='High Risk (PM2.5)')
plt.legend()
plt.title('PM2.5 Test Prediction')

plt.subplot(3, 1, 2)
plt.plot(df['RH'], label='Historical RH')
plt.plot(test_df.index[seq_length:], test_preds[:, 1], label='Test Predicted RH')
plt.axhline(y=60, color='r', linestyle='--', label='Moderate Risk (RH)')
plt.axhline(y=80, color='y', linestyle='--', label='High Risk (RH)')
plt.legend()
plt.title('Relative Humidity Test Prediction')

plt.subplot(3, 1, 3)
plt.plot(future_df['PM25'], label='Forecasted PM2.5')
plt.plot(future_df['RH'], label='Forecasted RH')
plt.axhline(y=35, color='r', linestyle='--', label='Moderate Risk (PM2.5)')
plt.axhline(y=55.4, color='y', linestyle='--', label='High Risk (PM2.5)')
plt.axhline(y=60, color='r', linestyle='--', label='Moderate Risk (RH)')
plt.axhline(y=80, color='y', linestyle='--', label='High Risk (RH)')
plt.legend()
plt.title('Future Prediction for Four Weeks')

plt.tight_layout()
plt.show()
