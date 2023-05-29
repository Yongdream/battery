import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply hidden_size by 2 for bidirectional LSTM

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(1), self.hidden_size).to(
            x.device)  # Multiply num_layers by 2 for bidirectional LSTM
        c0 = torch.zeros(self.num_layers * 2, x.size(1), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[-1, :, :])  # Take the last time step's output and pass it through the fully connected layer

        return out


class BiLSTMAdFeatures(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTMAdFeatures, self).__init__()

        self.lstm_model = BiLSTM(input_size, hidden_size, num_layers, output_size)
        self.fc = nn.Linear(output_size + additional_features_size, output_size)
        self.__in_features = 128

    def forward(self, x, additional_features):
        lstm_output = self.lstm_model(x)
        combined_input = torch.cat((lstm_output, additional_features), dim=1)
        output = self.fc(combined_input)

        return output

    def output_num(self):
        return self.__in_features
