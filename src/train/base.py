import torch
from torch.utils import data
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch import optim, nn
from pathlib import Path
from torch.optim import Adam
from collections import defaultdict

import copy
# import Utilities, Parametrization, Dataset, Utils
import time


class Base_trainer():
    """
    This is the base trainer for both the generator and reconstructor
    """

    def __init__(
            self,
            G_model,
            G_optimizer,
            train_batch_size=64,
            train_num_steps=10000,
            save_model=True,
            save_every=1000,
            loss_track_every=100,
            results_folder='./results'
    ):
        super().__init__()

        #         self.width = diffusion_model.width
        self.G = G_model
        #         self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        #         self.step_start_ema = step_start_ema
        self.save_every = save_every
        self.save_model = save_model

        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps

        self.G_optimizer = G_optimizer
        # self.train_lr_gamma = train_lr_gamma
        # self.train_lr_step_size = train_lr_step_size
        # self.G_scheduler = optim.lr_scheduler.StepLR(optimizer=self.G_optimizer, gamma=self.train_lr_gamma, step_size=self.train_lr_step_size)


        self.step = 0


        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # Track the loss at some time points for analysis
        self.loss_track_every = loss_track_every
        self.loss_tracker = defaultdict(list)

        # Track the best model
        self.best_G_step = 0
        self.best_G_loss = None
        self.best_G_model = self.G.state_dict()
        self.key = 0

    def save_G(self, milestone=1, model_type='generator'):
        data = {
            'step': self.best_G_step,
            'best_generator': self.best_G_model,
            'generator': self.G.state_dict(),
            'optimizer': self.G_optimizer.state_dict(),
            'loss': self.loss_tracker,
            'key': self.key
        }
        torch.save(data, str(self.results_folder / f'{model_type}-model-{milestone}.pt'))

    def load_G(self, milestone=1, model_type='generator'):
        data = torch.load(str(self.results_folder / f'{model_type}-model-{milestone}.pt'))

        self.step = data['step']
        self.G_model.load_state_dict(data['best_generator'])
        # self.G_model.load_state_dict(data['generator'])
        #         self.ema.load_state_dict(data['ema'])
        self.G_optimizer.load_state_dict(data['optimizer'])
        self.loss_tracker = data['loss']

    def evaluate(self, device):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

