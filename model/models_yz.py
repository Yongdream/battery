import torch.nn as nn
import torch
import torch.nn.functional as F


class yang(nn.Module):
    def __init__(self, channel=20, embed_dim=256, timewin=225):
        super(yang, self).__init__()
        self.embed_dim = embed_dim
        self.patch_embedding = nn.Linear(timewin, embed_dim)
        self.position_embedding = nn.parameter.Parameter(torch.FloatTensor(torch.randn([1, channel, embed_dim])))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.avg = nn.AvgPool1d(1)
        self.fc_1 = nn.Linear(embed_dim, 64)
        self.fc_2 = nn.Linear(64, 5)
        self.softmax = nn.Softmax(-1)
        self.Sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.act = nn.GELU()
        # self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.permute(0, 2, 1).float()
        x = self.patch_embedding(x)
        x = x + self.position_embedding  # [20, 256]

        x = self.avg(x)
        x = x.mean(dim=1)  # [B, 256]
        x = self.fc_1(x)  # [B, 64]
        x = self.act(x)  # [B, 64]
        x = self.fc_2(x)  # [B, 2]
        x = self.Sigmoid(x)  # [B, 2]
        return x


class transformer(nn.Module):
    def __init__(self, channel=20, embed_dim=256, timewin=225, num_heads=2, num_layers=2):
        super(transformer, self).__init__()
        self.embed_dim = embed_dim
        self.patch_embedding = nn.Linear(timewin, embed_dim)
        self.position_embedding = nn.parameter.Parameter(torch.FloatTensor(torch.randn([1, channel, embed_dim])))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads),
                                                 num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.avg = nn.AvgPool1d(1)
        self.fc_1 = nn.Linear(embed_dim, 64)
        self.fc_2 = nn.Linear(64, 3)
        self.softmax = nn.Softmax(-1)
        self.Sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.act = nn.GELU()
        # self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # [B, 225, 20]  [225, 20]
        B, T, P = x.size()
        x = x.permute(0, 2, 1).float()
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d',
        #                     b=b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = self.patch_embedding(x)

        x = x + self.position_embedding     # [20, 256]
        x = self.transformer(x)     # [B, 20, 256]
        x = self.avg(x)             # [B, 256, 1]
        x = x.mean(dim=1)   # [B, 256]
        x = self.fc_1(x)    # [B, 64]
        x = self.act(x)     # [B, 64]
        x = self.fc_2(x)    # [B, 2]
        x = self.Sigmoid(x)     # [B, 2]
        return x


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.patch_embedding = nn.Linear(225, 256)
        self.position_embedding = nn.Parameter(torch.FloatTensor(torch.randn([1, 20, 256])))
        # Position embedding

        self.conv1 = nn.Conv1d(20, 32, kernel_size=64, stride=16, padding=16)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)

        self.BN_1 = nn.BatchNorm1d(32, track_running_stats=False)
        self.BN_2 = nn.BatchNorm1d(64, track_running_stats=False)

        self.MaxPooling1D_1 = nn.MaxPool1d(kernel_size=2)
        self.MaxPooling1D_2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(p=0.3)  # Dropout layer 1
        self.dropout2 = nn.Dropout(p=0.3)  # Dropout layer 2

        self.fc = nn.Linear(64, 5)  # Classification layer

    def forward(self, x):
        x = x.permute(0, 2, 1).float()      # (B, 225, 20) Reshape to (B, 20, 225)
        x = self.patch_embedding(x)         # (B, 20, 256)
        x = x + self.position_embedding     # (B, 20, 256)

        x = self.conv1(x)                   # (B, 64, 15)
        # x = self.BN_1(x)
        x = F.relu(x)
        x = self.MaxPooling1D_1(x)          # (B, 64, 7)
        # x = self.dropout1(x)

        x = self.conv2(x)                   # (B, 32, 7)
        x = self.BN_2(x)
        x = F.relu(x)
        x = self.MaxPooling1D_2(x)          # (B, 32, 3)
        # x = self.dropout2(x)
        # x = F.relu(self.conv3(x))

        x = torch.mean(x, dim=2)    # Global average pooling  (B, 32)
        x = self.fc(x)              # Classification layer    (B, 5)

        return x