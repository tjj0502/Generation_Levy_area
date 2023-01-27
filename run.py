"""
Procedure for calibrating generative models using the unconditional Sig-Wasserstein metric.
"""
import ml_collections
import copy
import wandb
import yaml
import os
from os import path as pt

from src.evaluation.evaluations import full_evaluation
from src.utils import set_seed
import torch
import seaborn as sns
# from src.utils import get_experiment_dir, save_obj
# import argparse


def run(algo_id, config, base_dir, dataset):
    print('Executing: %s, %s' % (algo_id, dataset))
    experiment_directory = pt.join(base_dir, dataset, 'path_dim={}_D_batch_size={}_{}'.format(config.path_dim, config.D_batch_size, config.discriminator), 'seed={}'.format(config.seed), algo_id)

    print(experiment_directory)
    config.update(
        {"experiment_directory": experiment_directory}, allow_val_change=True)


    # initialize weight and bias
    # Place here your API key.
    # setup own api key in the config
    os.environ["WANDB_API_KEY"] = config.wandb_api
    tags = [
        algo_id,
        dataset,
    ]

    run = wandb.init(
        project='Levy area',
        config=copy.deepcopy(dict(config)),
        entity="jiajie0502",
        name="Grid test on batch size {}".format(config.D_batch_size),
        tags=tags,
        group=dataset,
        # name=config.algo,
        reinit=True
        # save_code=True,
        # job_type=config.function,
    )
    config = wandb.config
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)get_dataset


    # Define transforms and create dataloaders

    # WandB – wan
    # from src.datasets.dataloader import db.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    # wandb.watch(model, log="all", log_freq=200) # -> There was a wandb bug that made runs in Sweeps crash

    # Create model directory and instantiate config.path
    # get_experiment_dir(config)

    # Train the model
    if config.train:
        if not pt.exists(experiment_directory):
            # if the experiment directory does not exist we create the directory
            os.makedirs(experiment_directory)

        from src.train.trainer import get_trainer
        trainer = get_trainer(config, dataset)

        # Print arguments (Sanity check)
        print(config)
        # Train the model
        import datetime

        print(datetime.datetime.now())
        trainer.fit(config.device)

    elif config.pretrained:
        pass

    from src.data.dataloader import get_dataset
    train_dl, test_dl = get_dataset(config, dataset_name=dataset, num_workers=4)
    from src.model.generator.logsig_generator import Conditional_Logsig_Generator
    generator = Conditional_Logsig_Generator(input_dim=config.G_input_dim,
                                             hidden_dim=config.G_hidden_dim,
                                             path_dim=config.path_dim,
                                             logsig_level=config.logsig_level,
                                             device=config.device).to(config.device)

    generator.load_state_dict(torch.load(config.experiment_directory + '/generator-model-final.pt')['best_generator'])

    generator.eval()

    full_evaluation(generator, train_dl, test_dl, config)

    run.finish()

#     plot_summary(fake_test_dl, test_dl, config)
#     wandb.save(pt.join(config.exp_dir, '*png*'))
#     wandb.save(pt.join(config.exp_dir, '*pt*'))
#     wandb.save(pt.join(config.exp_dir, '*pdf*'))

def main(config):
    '''
    Main function, setup of the experiment.
    '''
    if not pt.exists('./data'):
        os.mkdir('./data')
    print('Start of training. CUDA: %s' % config.use_cuda)
    for dataset in config.datasets:
        config.update({"dataset": dataset}, allow_val_change=True)
        for algo_id in config.algo:
            for seed in range(config.initial_seed, config.initial_seed + config.num_seeds):
                # Set seed for exact reproducibility of the experiments
                set_seed(seed)
                config.update({"seed": seed}, allow_val_change=True)
                run(
                    algo_id=algo_id,
                    config=config,
                    dataset=dataset,
                    base_dir=config.base_dir,
                )



if __name__ == '__main__':
    sns.set()
    config_dir = 'configs/' + 'train_brownian.yaml'
    with open(config_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))

    # Cuda setup
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    if (config.use_cuda and torch.cuda.is_available()):
        print("Number of GPUs available: ", torch.cuda.device_count())
        config.update({"device": "cuda"}, allow_val_change=True)
    else:
        config.update({"device": "cpu"}, allow_val_change=True)

    # config.dataset = args.dataset
    key = int(torch.randint(1000,[1]))
    config.update({"key": key}, allow_val_change=True)
    
    D_batch_size = [20, 50, 100, 200, 500]
    for i in D_batch_size:
        key = int(torch.randint(1000,[1]))
        config.update({"key": key}, allow_val_change=True)
        config.update({"D_batch_size": i}, allow_val_change=True)
        # save_path = "./result/brownian/lr_g={}".format(i)
        # config.update({"results_folder": save_path}, allow_val_change=True)
        main(config)
