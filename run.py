"""
Procedure for calibrating generative models using the unconditional Sig-Wasserstein metric.
"""
import ml_collections
import copy
import wandb
import yaml
import os

from src.evaluation.evaluations import full_evaluation
import torch
# from src.utils import get_experiment_dir, save_obj
# import argparse


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # Set the seed
    torch.manual_seed(config.seed)
    # np.random.seed(config.seed)

    # initialize weight and bias
    # Place here your API key.
    # setup own api key in the config
    os.environ["WANDB_API_KEY"] = config.wandb_api
    tags = [
        config.algo,
        config.dataset,
    ]

    run = wandb.init(
        project='Levy area',
        config=copy.deepcopy(dict(config)),
        entity="jiajie0502",
        name="{}_lr={}".format(config.algo, config.lr_G),
        tags=tags,
        group=config.dataset,
        # name=config.algo,
        reinit=True
        # save_code=True,
        # job_type=config.function,
    )
    config = wandb.config
    print(config)
    if (config.device ==
            "cuda" and torch.cuda.is_available()):
        config.update({"device": "cuda:0"}, allow_val_change=True)
    else:
        config.update({"device": "cpu"}, allow_val_change=True)
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)get_dataset
    from src.train.trainer import get_trainer
    trainer = get_trainer(config)

    # Define transforms and create dataloaders

    # WandB – wan
    # from src.datasets.dataloader import db.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    # wandb.watch(model, log="all", log_freq=200) # -> There was a wandb bug that made runs in Sweeps crash

    # Create model directory and instantiate config.path
    # get_experiment_dir(config)

    # Train the model
    if config.train:
        # Print arguments (Sanity check)
        print(config)
        # Train the model
        import datetime

        print(datetime.datetime.now())
        trainer.fit(config.device)

    elif config.pretrained:
        pass
    
    from src.data.dataloader import get_dataset
    train_dl, test_dl = get_dataset(config, num_workers=4)
    from src.model.generator.logsig_generator import Base_Logsig_Generator
    generator = Base_Logsig_Generator(input_dim=config.G_input_dim,
                                             hidden_dim=config.G_hidden_dim,
                                             path_dim=config.path_dim,
                                             logsig_level=config.logsig_level,
                                             device=config.device).to(config.device)
    
    generator.load_state_dict(torch.load(config.results_folder + '/generator-model-final.pt')['best_generator'])
    
    generator.eval()
    
    full_evaluation(generator, train_dl, test_dl, config)
    
    run.finish()

#     plot_summary(fake_test_dl, test_dl, config)
#     wandb.save(pt.join(config.exp_dir, '*png*'))
#     wandb.save(pt.join(config.exp_dir, '*pt*'))
#     wandb.save(pt.join(config.exp_dir, '*pdf*'))


if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--algo', type=str, default='LSTM_DEV',
#                         help='choose from TimeGAN,RCGAN,TimeVAE')
#     parser.add_argument('--dataset', type=str, default='AR1',
#                         help='choose from AR1, ROUGH, GBM')
#     args = parser.parse_args()
#     if args.algo == 'TimeVAE':
#         config_dir = 'configs/' + 'train_vae.yaml'

#     else:
#         config_dir = 'configs/' + 'train_gan.yaml'
        
    config_dir = 'configs/' + 'train_brownian.yaml'
    with open(config_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    # config.dataset = args.dataset
    key = int(torch.randint(1000,[1]))
    config.update({"key": key}, allow_val_change=True)
    
    lr_g = [0.001]
    for i in lr_g:
        key = int(torch.randint(1000,[1]))
        config.update({"key": key}, allow_val_change=True)
        config.update({"lr_G": i}, allow_val_change=True)
        save_path = "./result/brownian/lr_g={}".format(i)
        config.update({"results_folder": save_path}, allow_val_change=True)
        main(config)
