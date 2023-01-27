from src.train.levy import Levy_GAN_trainer
from src.model.generator.logsig_generator import Conditional_Logsig_Generator
from src.model.discriminator.characteristic_discriminator import Grid_Characteristic_Discriminator, Gaussian_Characteristic_Discriminator, Embedded_Characteristic_Discriminator, IID_Gaussian_Characteristic_Discriminator, Cauchy_Characteristic_Discriminator
import torch

GENERATORS = {'brownian': Conditional_Logsig_Generator
              }

DISCRIMINATOR = {'grid_characteristic': Grid_Characteristic_Discriminator,
                 'cauchy_characteristic': Cauchy_Characteristic_Discriminator,
                 'gaussian_characteristic': Gaussian_Characteristic_Discriminator,
                 'iid_gaussian_characteristic': IID_Gaussian_Characteristic_Discriminator,
                 'embedded_characteristic': Embedded_Characteristic_Discriminator
                 }


def get_trainer(config, dataset):
    model_name = dataset

    generator = GENERATORS[config.generator](input_dim=config.G_input_dim,
                                             hidden_dim=config.G_hidden_dim,
                                             path_dim=config.path_dim,
                                             logsig_level=config.logsig_level,
                                             device=config.device)
    G_optimizer=torch.optim.Adam(
                generator.parameters(), lr=config.lr_G, betas=(0, 0.9))
    
    if config.discriminator == 'embedded_characteristic':
        discriminator = DISCRIMINATOR[config.discriminator](batch_size = config.D_batch_size,
                                                            hidden_dim = config.G_hidden_dim,
                                                            path_dim = config.path_dim)
    else:
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
