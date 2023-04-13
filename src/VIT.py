import torch
import torch.nn as nn
from einops import rearrange
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dwt import win_data, onehot_data
import numpy as np
import copy


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768, dropout=0.):
        super().__init__()
        n_patches = (image_size // patch_size) * (image_size // patch_size)
        self.patch_embedding = nn.Conv2d(in_channels=in_channels,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size)
        self.dropout = nn.Dropout(dropout)

        # TODO: add class token        ([1, 1, 768])
        self.class_token = nn.parameter.Parameter(torch.FloatTensor(torch.zeros([1, 1, embed_dim])))
        # default_initializer=nn.initializer.Constant(0.) dtype='float32'
        # TODO: add position embedding ([1, 197, 768])
        self.position_embedding = nn.parameter.Parameter(torch.FloatTensor(torch.randn([1, n_patches + 1, embed_dim])))
        # default_initializer=nn.initializer.TruncatedNormal(std=.02)  dtype='float32'

    def forward(self, x):
        # [n, c, h, w] ([4, 1, 768])
        class_tokens = self.class_token.expand([x.shape[0], -1, -1])  # for batch
        x = self.patch_embedding(x)  # [n, embed_dim, h', w']  ([4, 768, 14, 14])
        x = x.flatten(2)            # ([4, 768, 196])
        x = x.permute([0, 2, 1])    # ([4, 196, 768])
        x = torch.cat((class_tokens, x), 1)

        x = x + self.position_embedding
        return x


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim  # 768
        self.num_heads = num_heads  # 4
        self.head_dim = int(embed_dim / num_heads)  # 192
        self.all_head_dims = self.head_dim * self.num_heads     # 768
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_dims * 3) # [768, 768*3]
        self.proj = nn.Linear(self.all_head_dims, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(-1)


    def transpose_multi_head(self, x):
        # x ([4, 197, 768]) [B, num_patches, all_head_dims]
        new_shape = list(x.shape[:-1]) + [self.num_heads, self.head_dim]
        # [4, 197, 4, 192] [B, num_patches, num_heads, head_dim]
        x = x.reshape(new_shape)
        x = x.permute([0, 2, 1, 3])     # [4, 4, 197, 192]  1,2 交换位置是把每一个num_head进行处理
        return x

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, -1)
        q, k, v = map(self.transpose_multi_head, qkv)
        # q, k, v [4, 4, 197, 192]
        k = k.permute([0, 1, 3, 2])
        attn = torch.matmul(q, k)
        attn = self.scale * attn
        attn = self.softmax(attn)
        # ([4, 4, 197, 197])

        out = torch.matmul(attn, v)         # [4, 4, 197, 192]
        out = out.permute([0, 2, 1, 3])     # [4, 197, 4, 192]
        out = out.reshape([B, N, -1])       # [4, 197, 768]
        out = self.proj(out)                # [4, 197, 768]
        return out


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, qkv_bias=True, mlp_ratio=4.0, dropout=0., attention_dropout=0.):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)

    def forward(self, x):
        h = x  # residual
        x = self.attn_norm(x)
        x = self.attn(x)    # [4, 197, 768]
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, depth):
        super().__init__()
        layer_list = []
        for i in range(depth):
            encoder_layer = EncoderLayer()
            layer_list.append(encoder_layer)
        self.layers = nn.ModuleList(layer_list)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


# class VisualTransformer(nn.Module):
#     def __init__(self,
#                  image_size=224,
#                  patch_size=16,
#                  in_channels=3,
#                  num_classes=1000,
#                  embed_dim=768,
#                  depth=3,
#                  num_heads=8,
#                  mlp_ratio=4,
#                  qkv_bias=True,
#                  dropout=0.,
#                  attention_dropout=0.):
#         super().__init__()
#         self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
#         self.encoder = Encoder(embed_dim, depth)
#         self.classifier = nn.Linear(embed_dim, num_classes)
#
#     def forward(self, x):
#         # TODO: forward
#         # x:[N, C, H, W]
#         x = self.patch_embedding(x)  # [N, embed_dim, h', w']  ([4, 197, 768])
#         # print('-------------------------------------------------------')
#         # print(x.shape)
#         # x = x.flatten(2)  # [N, embed_dim, h'*w'] h'*w'=num_patches
#         # x = x.transpose([0, 2, 1])  # [N, num_patches, embed_dim]
#         print(x.shape)
#         x = self.encoder(x)
#         print(x.shape)
#         print(x[:, 0].shape)
#         x = self.classifier(x[:, 0])
#         return x

class transformer(nn.Module):
    def __init__(self, channel=20, embed_dim=256, timewin=225, num_heads=4, num_layers=1):
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
        self.act = nn.GELU()
    def forward(self, x):
        # [B, 225, 20]  [225, 20]
        print(x.shape)
        T, P = x.size()
        # x = x.reshape(-1, T, P)     # [1, 225, 20]
        x = x.reshape(T, P)     # [1, 225, 20]
        # x = x.permute(0, 2, 1)      # [B, 20, 225]  [20, 225]
        x = x.permute(1, 0)      # [B, 20, 225]  [20, 225]
        x = self.patch_embedding(x)  # [B, 20, 256]  [20, 256]
        x = x + self.position_embedding # [20, 256]
        x = self.transformer(x)     # [B, 20, 256]  [20, 256]
        x = self.avg(x)             # [B, 256, 1]
        x = self.fc_1(x)
        x = self.act(x)
        x = self.fc_2(x)
        x = self.softmax(x)
        return x

# class VisualTransformer(nn.Module):
#     def __init__(self, input_size, num_classes, image_size, patch_size, hidden_dim, num_layers, num_heads, dropout):
#         super().__init__()
#         self.input_size = input_size
#         self.num_classes = num_classes
#         self.patch_size = patch_size
#         self.image_size = image_size
#
#         self.embed_dim = hidden_dim * num_heads
#         self.num_patches = (image_size // patch_size) ** 2
#         self.patch_dim = self.input_size * self.patch_size
#
#         self.pos_embedding = nn.Parameter(torch.randn(1, 225, self.embed_dim))
#         self.patch_embedding = nn.Linear(self.patch_dim, self.embed_dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
#         self.dropout = nn.Dropout(dropout)
#
#         self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads),
#                                                 num_layers)
#         self.fc = nn.Linear(self.embed_dim, self.num_classes)
#
#     def forward(self, x):
#         T, P = x.shape()  # [B, 225, 20]  [225, 20]
#         x = self.patch_embedding(x)  # [B, 225, 20]  [225, 20]
#         x = x + self.pos_embedding()
#         x = x.permute(0, 2, 1)
#         x = self.transformer(x)

    # def forward(self, x):
    #     t, p = x.shape  # [15, 15, 20] torch.Size([225, 20])
    #     x = x.reshape(-1, t, p)
    #     # Reshape the input to a 2D image with patches
    #     # x = rearrange(x, 'b t (h p) -> b (t h) p', h=self.image_size // self.patch_size)
    #     # Add class token to input
    #     # cls_token = self.cls_token.expand(b, -1, -1)
    #     # print(cls_token.shape)
    #     # x = torch.cat([cls_token, x], dim=1)
    #     # Add position embeddings to input
    #     x = x + self.pos_embedding[:, :x.shape[1], :]
    #     # Pass through patch embeddings and dropout
    #     # x = self.patch_embedding(x)
    #     x = self.dropout(x)
    #     # Pass through transformer encoder
    #     x = x.transpose(0, 1)  # Transformer expects shape (sequence_length, batch_size, embedding_dim)
    #     x = self.transformer(x)
    #     x = x.transpose(0, 1)  # Transform back to shape (batch_size, sequence_length, embedding_dim)
    #     # Extract the first token, which represents the class token
    #     x = x[:, 0, :]
    #     # Pass through fully connected layer
    #     x = self.fc(x)
    #
    #     return x

# TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()

        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(p=dropout_rate)
        )
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attention_out = self.attention(x, x, x)[0]
        x = x + self.dropout1(attention_out)
        x = self.layer_norm1(x)
        mlp_out = self.mlp(x)
        x = x + self.dropout2(mlp_out)
        x = self.layer_norm2(x)
        return x


# 定义一个函数用于训练模型
def train(model, optimizer, criterion, dataloader, label, device, num_epochs):
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        running_loss = 0.0

        for batch in tqdm(dataloader, desc='Training'):
            print(type(batch))

            inputs = torch.from_numpy(batch).to(device)
            # labels = torch.onehot_data.to(device)

            print(f"inputs.shape:{inputs.shape}")
            print(f"inputs.dtype:{inputs.dtype}")
            optimizer.zero_grad()
            outputs = model(inputs)
            print(label.shape)
            loss = criterion(outputs, label)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(dataloader)
        print(f"Epoch{ epoch+1 }:\t Training Loss: {average_loss:.4f}")
        return average_loss


# 定义一个函数用于测试模型
def test(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc='Testing'):
            inputs = data.float().to(device)
            labels = torch.zeros(inputs.shape[0], dtype=torch.long).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels)

            running_loss += loss.item()

    accuracy = running_corrects.double() / len(dataloader.dataset)
    return running_loss / len(dataloader), accuracy



def main():
    # vit = VisualTransformer
    # 定义训练参数以及超参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = 20  # Number of features at each time point
    num_classes = 3
    image_size = 8  # Number of time points in each patch
    patch_size = 4
    hidden_dim = 64
    num_layers = 3
    num_heads = 4
    dropout = 0.1
    batch_size = 32
    lr = 1e-3
    num_epochs = 10

    # 将数据集划分为训练集和测试集
    train_size = int(0.8 * len(win_data))
    test_size = len(win_data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(win_data, [train_size, test_size])

    # model = vit(input_size, num_classes, image_size, patch_size, hidden_dim, num_layers, num_heads, dropout).to(device)
    model = transformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print('Model build successfully')

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)
    print(type(train_loader))

    dataloader = train_loader.dataset
    # 如果使用了Subset，则可以使用以下代码来提取其父数据集
    if isinstance(dataloader, torch.utils.data.Subset):
        dataloader_temp = dataloader.dataset

    labels = DataLoader(onehot_data, batch_size=10, shuffle=False)
    label = labels.dataset

    train(model, optimizer, criterion, dataloader, label, device, num_epochs)



if __name__ == "__main__":
    main()