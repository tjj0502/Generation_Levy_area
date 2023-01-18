import torch
from torch import nn
import signatory


class Characteristic_Discriminator(nn.Module):
    '''
    We parametrize a discrite random variable
    '''
    def __init__(self, batch_size: int, path_dim: int, logsig_level: int = 2):
        super(Characteristic_Discriminator, self).__init__()
        self.batch_size = batch_size
        self.path_dim = path_dim
        self.logsig_length = signatory.logsignature_channels(self.path_dim, logsig_level)


    def forward(self, fake_characteristic, real_characteristic, t=0.1):
        raise NotImplementedError

class Grid_Characteristic_Discriminator(Characteristic_Discriminator):
    '''
    We parametrize an empirical distribution on the coefficient space
    '''
    def __init__(self, batch_size: int, path_dim: int, logsig_level: int = 2):
        super(Grid_Characteristic_Discriminator, self).__init__(batch_size, path_dim, logsig_level)

        self.coefficients = nn.Parameter(torch.empty(self.batch_size, self.logsig_length))
        nn.init.kaiming_normal_(self.coefficients)


    def forward(self, fake_characteristic, real_characteristic, t=0.1):
        char_fake = fake_characteristic(self.coefficients, t)
        char_real = real_characteristic(self.coefficients, self.path_dim, t)
        D_loss = - 1 / self.batch_size * ((char_fake - char_real).real ** 2 + (char_fake - char_real).imag ** 2).sum()
        return D_loss, self.coefficients


class Gaussian_Characteristic_Discriminator(Characteristic_Discriminator):
    '''
    We assume the coefficients follows a Gaussian distribution where the mean and variance are optimized
    '''
    def __init__(self, batch_size: int, path_dim: int, logsig_level: int = 2):
        super(Gaussian_Characteristic_Discriminator, self).__init__(batch_size, path_dim, logsig_level)

        self.mean = nn.Parameter(torch.empty(1, self.logsig_length))
        self.logvar = nn.Parameter(torch.empty(1, self.logsig_length))
        nn.init.kaiming_normal_(self.logvar)


    def forward(self, fake_characteristic, real_characteristic, t=0.1):
        coefficients = self.mean + torch.pow(torch.exp(self.logvar), 0.5) * torch.randn([self.batch_size, self.logsig_length]).to(self.logvar.device)
        char_fake = fake_characteristic(coefficients, t)
        char_real = real_characteristic(coefficients, self.path_dim, t)
        D_loss = - 1 / self.batch_size * ((char_fake - char_real).real ** 2 + (char_fake - char_real).imag ** 2).sum()
        return D_loss, self.logvar

class Embedded_Characteristic_Discriminator(Characteristic_Discriminator):
    '''
    We assume the coefficients follows a Gaussian distribution where the mean and variance are optimized
    '''
    def __init__(self, batch_size: int, path_dim: int, hidden_dim: int, logsig_level: int = 2):
        super(Embedded_Characteristic_Discriminator, self).__init__(batch_size, path_dim, logsig_level)

        self.hidden_dim = hidden_dim
        self.mean = nn.Parameter(torch.empty(1, self.logsig_length))
        self.logvar = nn.Parameter(torch.empty(1, self.logsig_length))
        nn.init.kaiming_normal_(self.logvar)

        self.model = nn.Sequential(
            *self.block(self.logsig_length, self.hidden_dim, normalize=False, activation='sigmoid'),
            *self.block(self.hidden_dim, int(self.logsig_length), normalize=False, activation='none')
        )


    def block(self, in_feat, out_feat, normalize = True, activation = 'sigmoid'):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat))
        if activation == 'relu':
            layers.append(nn.LeakyReLU(0.3))
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        return layers


    def forward(self, fake_characteristic, real_characteristic, t=0.1):
        coefficients = self.mean + torch.pow(torch.exp(self.logvar), 0.5) * torch.randn([self.batch_size, self.logsig_length]).to(self.logvar.device)
        coefficients = self.model(coefficients)
        char_fake = fake_characteristic(coefficients, t)
        char_real = real_characteristic(coefficients, self.path_dim, t)
        D_loss = - 1 / self.batch_size * ((char_fake - char_real).real ** 2 + (char_fake - char_real).imag ** 2).sum()
        return D_loss, self.logvar
    


