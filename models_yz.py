import torch.nn as nn
import torch


class transformer(nn.Module):
    def __init__(self, channel=20, embed_dim=256, timewin=225, num_heads=2, num_layers=4):
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
