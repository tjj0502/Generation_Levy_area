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
        self.latten_dim = 10
        # self.upperbound = 5
        # self.lowerbound = 5
        # self.register_buffer(name='coefficients', tensor=self.lowerbound + (self.upperbound-self.lowerbound)*torch.rand([self.batch_size, self.logsig_length]))
        # The set of all gammas and lambdas
        # self.coefficients = nn.Parameter(torch.empty(self.batch_size, self.logsig_length))
        
#         self.mean = nn.Parameter(torch.empty(self.logsig_length))
        self.logvar = nn.Parameter(torch.empty(1, self.logsig_length))
        # self.lambdas = nn.Parameter(torch.empty(self.batch_size, int(self.bm_dim*(self.bm_dim-1)/2)))
        nn.init.kaiming_normal_(self.logvar)
        # nn.init.normal_(self.logvar)
        
        # self.coefficients = torch.pow(torch.exp(self.logvar), 0.5) * torch.randn([self.batch_size, self.logsig_length]).to(self.logvar.device)
        # nn.init.kaiming_normal_(self.std)
        # nn.init.kaiming_normal_(self.lambdas)
        
        
        # Grid search
        # x = torch.arange(-1,1.1,0.1)
        # grid = torch.zeros([21**3, 3])
        # n = x.shape[0]
        # t = 0
        # for i in x:
        #     for j in x:
        #         for k in x:
        #             grid[t,0] = i
        #             grid[t,1] = j
        #             grid[t,2] = k
        #             t+=1
        # self.register_buffer(name='grid', tensor=grid)

    def forward(self, fake_characteristic, real_characteristic, t = 0.1):
        
        coefficients = torch.pow(torch.exp(self.logvar), 0.5) * torch.randn([self.batch_size, self.logsig_length]).to(self.logvar.device)
        # coefficients = self.grid
        
        # self.coefficients = self.lowerbound + (self.upperbound-self.lowerbound)*torch.rand([self.batch_size, self.logsig_length]).to(self.coefficients.device)
        char_fake = fake_characteristic(coefficients, t)
        char_real = real_characteristic(coefficients, self.path_dim, t)
        # D_loss = -torch.pow(char_fake - char_real, 2).sum() X * X^+
        # print('batch_size', self.batch_size, 'lambdas', (char_fake - char_real)[:3])
        D_loss = - 1/self.batch_size * ((char_fake - char_real).real**2 + (char_fake - char_real).imag**2).sum()
        return D_loss
    


