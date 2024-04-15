import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

class MLP_for_DeepFake(nn.Module):
    def __init__(self, input_features, dropout_rate, hidden_units):
        super(MLP_for_DeepFake, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_units[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        self.act1 = nn.ReLU()

        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(hidden_units)):
            self.hidden_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            self.hidden_layers.append(nn.Dropout(dropout_rate))
            self.hidden_layers.append(nn.ReLU())

        # Output layer for binary classification with 1 output unit
        self.output = nn.Linear(hidden_units[-1], 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        
        x = self.dropout1(self.act1(self.fc1(x)))
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
            else:
                x = layer(x)

        x = self.output(x)
        return x

class CNN_for_DeepFake(nn.Module):
    def __init__(self, dropout_rate, fc_units):
        super(CNN_for_DeepFake, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  
        self.act1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)  
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)   

        self.fc1 = nn.Linear(in_features=64 * 56 * 56, out_features=fc_units)
        self.fc2 = nn.Linear(fc_units, 1)
        self.dropout3 = nn.Dropout(dropout_rate)


    def forward(self, x):
        x = self.pool(self.bn1(self.act1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(self.bn2(self.act2(self.conv2(x))))
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.dropout3(self.fc1(x))
        x = self.fc2(x)

        return x
    
class CNN_LSTM_for_DeepFake(nn.Module):
    def __init__(self, dropout_rate, fc_units, lstm_units, num_layers):
        super(CNN_LSTM_for_DeepFake, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer
        
        # Calculate proper input dimensions
        # Assuming the output of the last pool layer is (64, 56, 56) based on calculations
        self.lstm = nn.LSTM(input_size=64 * 56, hidden_size=lstm_units, num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(lstm_units, 1)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool1(self.bn1(self.act1(self.conv1(x))))
        x = self.pool2(self.bn2(self.act2(self.conv2(x))))
        
        # Reshape for LSTM
        x = x.permute(0, 2, 3, 1).contiguous()  # Change dimension ordering
        x = x.view(x.size(0), x.size(1), -1)  # Flatten width and height into sequence length

        x, (hn, cn) = self.lstm(x)
        x = self.dropout3(x[:, -1, :])  # Taking the last sequence output

        x = self.fc(x)
        
        return x

    
