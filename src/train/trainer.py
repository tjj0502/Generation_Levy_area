from src.train.levy import Levy_GAN_trainer
from src.model.generator.logsig_generator import Base_Logsig_Generator
from src.model.discriminator.characteristic_discriminator import Characteristic_Discriminator
import torch

GENERATORS = {'brownian': Base_Logsig_Generator
              }

DISCRIMINATOR = {'characteristic': Characteristic_Discriminator
                 }


def get_trainer(config):
    model_name = config.dataset

    generator = GENERATORS[config.generator](input_dim=config.G_input_dim,
                                             hidden_dim=config.G_hidden_dim,
                                             path_dim=config.path_dim,
                                             logsig_level=config.logsig_level,
                                             device=config.device)
    G_optimizer=torch.optim.Adam(
                generator.parameters(), lr=config.lr_G, betas=(0, 0.9))
    
    discriminator = DISCRIMINATOR[config.discriminator](batch_size = config.D_batch_size,
                                                        path_dim = config.path_dim)
    
    D_optimizer=torch.optim.Adam(
                discriminator.parameters(), lr=config.lr_D, betas=(0, 0.9))
    # D_optimizer = None

    trainer = {
        "brownian": Levy_GAN_trainer(G=generator,
                                     G_optimizer=G_optimizer,
                                     D=discriminator,
                                     D_optimizer=D_optimizer,
                                     config=config)
    }[model_name]

    return trainer
