from torch import nn
import torch
import numpy as np
import signatory
# import Utilities, Parametrization


# GAN models
class Base_Logsig_Generator(nn.Module):
    '''
    Signature generator using logsig trivialization, conditional on the path length
    '''
    def __init__(self, input_dim, hidden_dim, path_dim, logsig_level, device):
        super(Base_Logsig_Generator, self).__init__()
        self.input_dim = input_dim  # Noise dimension
        self.path_dim = path_dim  # Time series dimension
        self.logsig_level = logsig_level  # Signature degree
        self.hidden_dim = hidden_dim  # Hidden dimension
        # self.aug_dim = 2 * self.dim  # Lead-lag augmentation
        self.logsig_length = torch.zeros(self.logsig_level)
        for i in range(1, self.logsig_level + 1):
            self.logsig_length[i - 1] = signatory.logsignature_channels(self.path_dim, i)
    
        self.model = nn.Sequential(
            *self.block(self.input_dim, self.hidden_dim, normalize=False, activation = 'sigmoid'),
            *self.block(self.hidden_dim , self.hidden_dim, activation='sigmoid'),
            *self.block(self.hidden_dim , int(self.logsig_length[-1]), normalize=False, activation='none')
        )
        
        # self.model_1 = nn.Sequential(
        #     *self.block(self.input_dim, self.hidden_dim, normalize=False),
        #     *self.block(self.hidden_dim , self.hidden_dim, activation='relu'),
        #     *self.block(self.hidden_dim , int(self.logsig_length[-1]) - int(self.logsig_length[0]), normalize=False, activation='none')
        # )
        
        for param in self.parameters():
            nn.init.normal_(param)
        
        self.device = device
    
    def block(self, in_feat, out_feat, normalize = True, activation = 'sigmoid'):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat))
        if activation == 'relu':
            layers.append(nn.LeakyReLU(0.3))
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        return layers

    def forward(self, z):
        self.gen_logsig = self.model(z)
        # gen_logsig_2 = self.model_1(z)
        # self.gen_logsig = torch.cat([gen_logsig_1, gen_logsig_2], -1)
        return self.gen_logsig


class Conditional_Logsig_Generator(nn.Module):
    '''
    The conditional generator aims to learn the conditional distribution of Levy area given the BM increment
    The model takes as input BM upto time T and outputs the corresponding Levy terms
    '''

    def __init__(self, input_dim, hidden_dim, path_dim, logsig_level, device):
        super(Conditional_Logsig_Generator, self).__init__()
        self.input_dim = input_dim  # Noise dimension
        self.path_dim = path_dim  # Time series dimension
        self.logsig_level = logsig_level  # Signature degree
        self.hidden_dim = hidden_dim  # Hidden dimension
        # self.aug_dim = 2 * self.dim  # Lead-lag augmentation
        self.logsig_length = torch.zeros(self.logsig_level)
        for i in range(1, self.logsig_level + 1):
            self.logsig_length[i - 1] = signatory.logsignature_channels(self.path_dim, i)

        self.model = nn.Sequential(
            *self.block(self.input_dim + self.path_dim, self.hidden_dim, normalize=False, activation='sigmoid'),
            *self.block(self.hidden_dim, self.hidden_dim, activation='sigmoid'),
            *self.block(self.hidden_dim, int(self.logsig_length[-1] - self.logsig_length[-2]), normalize=True, activation='none')
        )

        # self.model_1 = nn.Sequential(
        #     *self.block(self.input_dim, self.hidden_dim, normalize=False),
        #     *self.block(self.hidden_dim , self.hidden_dim, activation='relu'),
        #     *self.block(self.hidden_dim , int(self.logsig_length[-1]) - int(self.logsig_length[0]), normalize=False, activation='none')
        # )

        for param in self.parameters():
            nn.init.normal_(param)

        self.device = device

    def block(self, in_feat, out_feat, normalize=True, activation='sigmoid'):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat))
        if activation == 'relu':
            layers.append(nn.LeakyReLU(0.3))
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        return layers

    def forward(self, bm, z):
        # Generate logsignature of degree two
        gen_logsig = self.model(torch.cat([bm, z], -1))
        # gen_logsig_2 = self.model_1(z)
        gen_logsig = torch.cat([bm, gen_logsig], -1)
        return gen_logsig


class Brownian_Logsig_Generator(Base_Logsig_Generator):
    '''
    Logsignature generator
    '''

    def __init__(self, input_dim, hidden_dim, path_dim, device):
        super(Brownian_Logsig_Generator, self).__init__(input_dim, hidden_dim, path_dim, device)



        # Different models
        self.model = nn.Sequential(
            *self.block(self.input_dim, self.hidden_dim , normalize=False),
            *self.block(self.hidden_dim , self.hidden_dim ),
            *self.block(self.hidden_dim , int(self.logsig_length[-1]))
        )

    def forward(self, z):
        self.gen_logsig = self.model(z)
        return self.gen_logsig

