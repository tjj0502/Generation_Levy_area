import torch
from tqdm.auto import tqdm
import wandb
import copy
import math
from scipy.stats import special_ortho_group
from src.train.base import Base_trainer
from src.train.levy_characteristic import get_real_characteristic, get_fake_characteristic
from src.utils import toggle_grad
from src.utils import loader_to_tensor
from src.evaluation.test_metrics import HistoLoss, CovLoss
from src.data.dataloader import get_dataset
from src.evaluation.evaluations import fake_loader


class Levy_GAN_trainer(Base_trainer):
    def __init__(
            self,
            G,
            G_optimizer,
            D,
            D_optimizer,
            config
    ):
        super(Levy_GAN_trainer, self).__init__(G,
                                         G_optimizer,
                                         train_batch_size=config.train_batch_size,
                                         train_num_steps=config.train_num_steps,
                                         save_model=config.save_model,
                                         save_every=config.save_every,
                                         loss_track_every=config.loss_track_every,
                                         results_folder=config.experiment_directory)

        
        self.D = D
        self.D_optimizer = D_optimizer
        self.D_steps_per_G_step = config.D_steps_per_G_step
        self.G_steps_per_D_step = config.G_steps_per_D_step
        self.config = config
        self.T = config.T
        self.key = config.key

    def save_D(self, milestone=1, model_type='discriminator'):
        data = {
            'discriminator': copy.deepcopy(self.D.state_dict()),
            'key': self.key
        }
        torch.save(data, str(self.results_folder / f'{model_type}-model-{milestone}.pt'))

    def fit(self, device):
        self.G.to(device)
        self.D.to(device)
        # self.D.coefficients.to(device)
        # print(self.D.coefficients.device)
        real_characteristic = get_real_characteristic()
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                D_loss, G_loss = self.step_fit(real_characteristic)
                self.step +=1
                pbar.set_description('G_loss: {}, D_loss: {}'.format(G_loss, D_loss))
                torch.cuda.empty_cache()
                pbar.update(1)
            self.save_G('final')
            self.save_D('final')
            print('training complete')

    
    # def step_fit(self, real_characteristic):
    #     # Generate fake data for discriminator training
    #     toggle_grad(self.G, False)
    #     # with torch.no_grad():
    #     noise = torch.randn([self.batch_size, self.G.input_dim]).to(self.G.device)
    #     x_fake = self.G(noise)
    #     # bm_fake, levy_fake = x_fake[:,:self.dim], x_fake[:,self.dim:]
    #     # print(bm_fake.shape,levy_fake.shape)
    #     fake_characteristic = get_fake_characteristic(x_fake)
    #     for i in range(self.D_steps_per_G_step):    
    #         D_loss = self.D_trainstep(fake_characteristic, real_characteristic)
    #     if self.step != 0 and self.step % self.loss_track_every == 0:
    #         self.loss_tracker['D_loss'].append(D_loss)
    #         # wandb.log({'D_loss': D_loss}, step)
    #     x_fake.requires_grad_()
    #     G_loss = self.G_trainstep(fake_characteristic, real_characteristic)
    #     # wandb.log({'G_loss': G_loss}, step)
    #     return D_loss, G_loss
    
    
    def step_fit(self, real_characteristic):
        '''
        At each iteration, we first train the discriminator once and then train the generation multiple times
        Args:
            real_characteristic: function, characteristic function via analytical formula
        Return:
            D_loss: torch.tensor, discriminator loss
            G_loss: torch.tensor, generator loss
        '''
        with torch.no_grad():
            # Generate fake data for discriminator training
            noise = torch.randn([self.batch_size, self.G.input_dim]).to(self.G.device)
            BM_increment = math.sqrt(self.T) * torch.randn([self.batch_size, self.G.path_dim, 1]).to(self.G.device)
            # To give the rotational invariance, we randomly simulate a rotational matrix under the Haar measure and apply it to the BM increment.
            # rotation_matrix = torch.from_numpy(special_ortho_group.rvs(self.G.path_dim)).to(device = BM_increment.device, dtype = torch.float)
            # BM_increment = rotation_matrix @ BM_increment
            
            x_fake = self.G(BM_increment.squeeze(-1), noise)
            fake_characteristic = get_fake_characteristic(x_fake)
        D_loss = self.D_trainstep(fake_characteristic, real_characteristic)
        if self.step != 0 and self.step % self.loss_track_every == 0:
            self.loss_tracker['D_loss'].append(D_loss)
        wandb.log({'D_loss': D_loss}, self.step)
        for i in range(self.G_steps_per_D_step):  
            G_loss = self.G_trainstep(real_characteristic)
        
            wandb.log({'G_loss': G_loss}, self.step)
        
        return D_loss, G_loss
        
    
    def G_trainstep(self, real_characteristic):
        toggle_grad(self.G, True)
        self.G.train()
        noise = torch.randn([self.batch_size, self.G.input_dim]).to(self.G.device)
        BM_increment = math.sqrt(self.T) * torch.randn([self.batch_size, self.G.path_dim, 1]).to(self.G.device)
        # rotation_matrix = torch.from_numpy(special_ortho_group.rvs(self.G.path_dim)).to(device = BM_increment.device, dtype = torch.float)
        # BM_increment = rotation_matrix @ BM_increment
        x_fake = self.G(BM_increment.squeeze(-1), noise)
        fake_characteristic = get_fake_characteristic(x_fake)
        coefficients = self.D.mean + torch.pow(torch.exp(self.D.logvar), 0.5) * torch.randn([self.D.batch_size, self.D.logsig_length]).to(self.G.device) + 1e-5
        # coefficients = self.D.model(coefficients)
        char_fake = fake_characteristic(coefficients, self.T)
        char_real = real_characteristic(coefficients, self.G.path_dim, self.T)
        # G_loss = torch.pow(char_fake - char_real, 2).sum()
        G_loss = 1/self.D.batch_size * ((char_fake - char_real).real**2 + (char_fake - char_real).imag**2).sum()
        
        
        # print(G_loss)
        G_loss.backward()
        self.G_optimizer.step()

        if self.save_model and self.step != 0 and self.step % self.save_every == 0:
            milestone = self.step // self.save_every
            self.save_G(milestone)

        if self.step != 0 and self.step % self.loss_track_every == 0:
            with torch.no_grad():
                
                self.G.eval()
                train_dl, test_dl = get_dataset(self.config, dataset_name=self.config.dataset, num_workers=4)
                real_data = torch.cat([loader_to_tensor(train_dl), loader_to_tensor(test_dl)])
                fake_data = loader_to_tensor(fake_loader(self.G, num_samples=10000, batch_size=128, config=self.config))
                hist_loss = HistoLoss(real_data.unsqueeze(1), n_bins=50, name='marginal_distribution')(fake_data.unsqueeze(1))
                cov_loss = CovLoss(real_data.unsqueeze(1), name='covariance')(fake_data.unsqueeze(1))
                evaluation_loss = hist_loss + cov_loss
                wandb.log({'hist_loss': hist_loss})
                wandb.log({'cov_loss': cov_loss})
                # print("hist_loss and cov_loss logged! ", self.step// self.loss_track_every)
            self.loss_tracker['hist_loss'] = hist_loss
            self.loss_tracker['cov_loss'] = cov_loss
            
            if self.best_G_loss is None or (evaluation_loss) < self.best_G_loss:
                self.best_G_model = copy.deepcopy(self.G.state_dict())
                self.best_G_loss = evaluation_loss.clone()
                self.best_G_step = self.step
                print('updated: ', self.step, hist_loss, cov_loss)
            
            self.loss_tracker['best_G_loss'].append(self.best_G_loss)
            self.loss_tracker['evaluation_loss'].append(evaluation_loss)
            self.loss_tracker['G_loss'].append(G_loss)
            
            # Track other qualities for debugging
            self.loss_tracker['logvar'].append(self.D.logvar.clone())
            # wandb.log({'logvar': self.D.logvar})
        
        toggle_grad(self.G, False)

        return G_loss.item()

    def D_trainstep(self, fake_characteristic, real_characteristic):
        # Use x_fake to construct the joint characteristic function under fake empirical measure
        toggle_grad(self.D, True)
        
        self.D.train()
        self.D_optimizer.zero_grad()
        # On fake data
        # x_fake.requires_grad_()
        D_loss, _ = self.D(fake_characteristic, real_characteristic, self.T)
        D_loss.backward()
        self.D_optimizer.step()
        
        if self.save_model and self.step != 0 and self.step % self.save_every == 0:
            milestone = self.step // self.save_every
            self.save_D(milestone)
        
        toggle_grad(self.D, False)
        return D_loss

    

