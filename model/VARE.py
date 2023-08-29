import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from torchinfo import summary


class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param block: LSTM/GRU block
    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, block = 'LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, bidirectional=True)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, bidirectional=True)
        else:
            raise NotImplementedError

        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=hidden_size, out_features=180),
        )
        self.att_net.add_module("att_act", nn.ReLU())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=180, out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, x):
        """
        Forward propagation of encoder. Given input, outputs the last hidden state of encoder
        """
        x = x.permute(2, 0, 1)
        # torch.Size([256, 128, 16])
        _, h = self.model(x)
        # torch.Size([1, 128, 32])

        h_end = h[-1, :, :]     # torch.Size([128, 32])
        attention_score = self.att_net(h_end)           # torch.Size([128, 1])
        h_out_att = torch.mul(h_end, attention_score)   # 使用注意力分数加权 torch.Size([128, 90])
        # h_cat = torch.sum(h_out_att, dim=1)
        return h_out_att


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        # torch.Size([128, 32])
        self.latent_mean = self.hidden_to_mean(cell_output)
        # torch.Size([128, 20])
        self.latent_logvar = self.hidden_to_logvar(cell_output)
        # torch.Size([128, 20])

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)   # 计算潜在变量的标准差
            eps = torch.randn_like(std)                 # 引入噪音
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean


class Decoder(nn.Module):
    """Converts latent vector into output

    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype, block='GRU'):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype

        if block == 'LSTM':
            self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth)
        elif block == 'GRU':
            self.model = nn.GRU(1, self.hidden_size, self.hidden_layer_depth)       # 后期看能不能改为双向！
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

        self.decoder_inputs = torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True).type(self.dtype)
        self.c_0 = torch.zeros(self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).type(self.dtype)

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent):
        """Converts latent to hidden to output

        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        h_state = self.latent_to_hidden(latent)  # torch.Size([128, 32])
        device = h_state.device  # 获取h_state所在的设备

        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state.to(device) for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs.to(device), (h_0, self.c_0))
        elif isinstance(self.model, nn.GRU):
            h_0 = torch.stack([h_state.to(device) for _ in range(self.hidden_layer_depth)])  # torch.Size([5, 128, 32])
            decoder_output, _ = self.model(self.decoder_inputs.to(device), h_0)  # torch.Size([256, 128, 32])
        else:
            raise NotImplementedError

        out = self.hidden_to_output(decoder_output)         # torch.Size([256, 128, 16])
        out = out.permute(1, 2, 0)                          # torch.Size([128, 16, 256])
        return out


class Linear(nn.Module):
    def __init__(self, latent_length):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(latent_length, 64)
        self.fc2 = nn.Linear(64, 128)

    def forward(self, latent):
        x = self.fc1(latent)  # 第一层前向传播
        x = torch.relu(x)  # 使用ReLU激活函数
        x = self.fc2(x)  # 第二层前向传播
        return x


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


class VAREAdFeatures(nn.Module):
    """Variational recurrent auto-encoder. This module is used for dimensionality reduction of timeseries

    :param sequence_length: length of the input sequence
    :param number_of_features: number of input features
    :param hidden_size:  hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param batch_size: number of timeseries in a single batch
    :param learning_rate: the learning rate of the module
    :param block: GRU/LSTM to be used as a basic building block
    :param n_epochs: Number of iterations/epochs
    :param dropout_rate: The probability of a node being dropped-out
    :param optimizer: ADAM/ SGD optimizer to reduce the loss function
    :param loss: SmoothL1Loss / MSELoss / ReconLoss / any custom loss which inherits from `_Loss` class
    :param boolean cuda: to be run on GPU or not
    :param print_every: The number of iterations after which loss should be printed
    :param boolean clip: Gradient clipping to overcome explosion
    :param max_grad_norm: The grad-norm to be clipped
    :param dload: Download directory where models are to be dumped
    """
    def __init__(self, sequence_length, number_of_features, hidden_size=90, hidden_layer_depth=5, latent_length=20,
                 block='GRU', clip=True, max_grad_norm=5, dload='.'):

        super(VAREAdFeatures, self).__init__()

        self.dtype = torch.FloatTensor

        self.encoder = Encoder(number_of_features = number_of_features,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               block=block)

        self.lmbd = Lambda(hidden_size=hidden_size,
                           latent_length=latent_length)

        self.decoder = Decoder(sequence_length=sequence_length,
                               batch_size = batch_size,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               output_size=number_of_features,
                               block=block,
                               dtype=self.dtype)

        self.fc = Linear(latent_length=latent_length,
                         )

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        self.clip = clip
        self.max_grad_norm = max_grad_norm
        self.is_fitted = False
        self.dload = dload

        self.__in_features = 128

    def forward(self, x):
        """
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder

        :param x:input tensor
        :return: the decoded output, latent vector
        """
        cell_output = self.encoder(x)       # torch.Size([B, 90])
        latent = self.lmbd(cell_output)     # torch.Size([B, 20])
        features = self.fc(latent)
        decoded = self.decoder(latent)      # torch.Size([256, B, 16])

        return features

    def _rec(self, x_decoded, x, loss_fn):
        """
        Compute the loss given output x decoded, input x and the specified loss function

        :param x_decoded: output of the decoder
        :param x: input to the encoder
        :param loss_fn: loss function specified
        :return: joint loss, reconstruction loss and kl-divergence loss
        """
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar

        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        recon_loss = loss_fn(x_decoded, x)

        return kl_loss + recon_loss, recon_loss, kl_loss

    def compute_loss(self, X):
        """
        Given input tensor, forward propagate, compute the loss, and backward propagate.
        Represents the lifecycle of a single iteration

        :param X: Input tensor
        :return: total loss, reconstruction loss, kl-divergence loss and original input
        """
        x = Variable(X[:,:,:].type(self.dtype), requires_grad = True)

        x_decoded, _ = self(x)
        

        return loss, recon_loss, kl_loss, x

    def _train(self, train_loader):
        """
        For each epoch, given the batch_size, run this function batch_size * num_of_batches number of times

        :param train_loader:input train loader with shuffle
        :return:
        """
        self.train()

        epoch_loss = 0
        t = 0

        for t, X in enumerate(train_loader):

            # Index first element of array to return tensor
            X = X[0]

            # required to swap axes, since dataloader gives output in (batch_size x seq_len x num_of_features)
            X = X.permute(1,0,2)

            self.optimizer.zero_grad()
            loss, recon_loss, kl_loss, _ = self.compute_loss(X)
            loss.backward()

            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.max_grad_norm)

            # accumulator
            epoch_loss += loss.item()

            self.optimizer.step()

            if (t + 1) % self.print_every == 0:
                print('Batch %d, loss = %.4f, recon_loss = %.4f, kl_loss = %.4f' % (t + 1, loss.item(),
                                                                                    recon_loss.item(), kl_loss.item()))

        print('Average loss: {:.4f}'.format(epoch_loss / t))

    def _batch_transform(self, x):
        """
        Passes the given input tensor into encoder and lambda function

        :param x: input batch tensor
        :return: intermediate latent vector
        """
        return self.lmbd(
                    self.encoder(
                        Variable(x.type(self.dtype), requires_grad = False)
                    )
        ).cpu().data.numpy()

    def _batch_reconstruct(self, x):
        """
        Passes the given input tensor into encoder, lambda and decoder function

        :param x: input batch tensor
        :return: reconstructed output tensor
        """

        x = Variable(x.type(self.dtype), requires_grad = False)
        x_decoded, _ = self(x)

        return x_decoded.cpu().data.numpy()

    def reconstruct(self, dataset, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_reconstruct`
        Prerequisite is that model has to be fit

        :param dataset: input dataset who's output vectors are to be obtained
        :param bool save: If true, dumps the output vector dataframe as a pickle file
        :return:
        """

        self.eval()

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last=True) # Don't shuffle for test_loader

        if self.is_fitted:
            with torch.no_grad():
                x_decoded = []

                for t, x in enumerate(test_loader):
                    x = x[0]
                    x = x.permute(1, 0, 2)

                    x_decoded_each = self._batch_reconstruct(x)
                    x_decoded.append(x_decoded_each)

                x_decoded = np.concatenate(x_decoded, axis=1)

                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    x_decoded.dump(self.dload + '/z_run.pkl')
                return x_decoded

        raise RuntimeError('Model needs to be fit')

    def transform(self, dataset, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_transform`
        Prerequisite is that model has to be fit

        :param dataset: input dataset who's latent vectors are to be obtained
        :param bool save: If true, dumps the latent vector dataframe as a pickle file
        :return:
        """
        self.eval()

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last=True) # Don't shuffle for test_loader
        if self.is_fitted:
            with torch.no_grad():
                z_run = []

                for t, x in enumerate(test_loader):
                    x = x[0]
                    x = x.permute(1, 0, 2)

                    z_run_each = self._batch_transform(x)
                    z_run.append(z_run_each)

                z_run = np.concatenate(z_run, axis=0)
                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    z_run.dump(self.dload + '/z_run.pkl')
                return z_run

        raise RuntimeError('Model needs to be fit')

class VAREAPar(nn.Module):
    def __init__(self):
        super(VAREAPar, self).__init__()

        self.__in_features = 128
        
    def output_num(self):
        return self.__in_features



print_every = 30
clip = True     # options: True, False
max_grad_norm = 5
loss = 'MSELoss'    # options: SmoothL1Loss, MSELoss

hidden_size = 32
hidden_layer_depth = 5
latent_length = 20
block = 'GRU'  # options: LSTM, GRU

batch_size = 128
number_of_features = 16
sequence_length = 256

# (sequence_length, batch_size, number_of_features)
# (batch_size, number_of_features, sequence_length)
input_tensor = torch.randn(batch_size, number_of_features, sequence_length)

# 创建模型实例
model = VAREAdFeatures(sequence_length=sequence_length,
                        number_of_features=number_of_features,
                        hidden_size=hidden_size,
                        hidden_layer_depth=hidden_layer_depth,
                        latent_length=latent_length,
                        clip=clip,
                        max_grad_norm=max_grad_norm,
                        block=block)


features = model(input_tensor)

# summary(model, input_size=(256, batch_size, 16))

# 打印输出张量的形状
print("features shape:", features.shape)

