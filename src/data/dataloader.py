import torch
import ml_collections
from typing import Tuple
from torch.utils.data import DataLoader
from src.data.BM import BM


def get_dataset(
    config: ml_collections.ConfigDict,
    num_workers: int = 4,
    data_root="./data",
) -> Tuple[dict, torch.utils.data.DataLoader]:
    """
    Create datasets loaders for the chosen datasets
    :return: Tuple ( dict(train_loader, val_loader) , test_loader)
    """
    dataset = {
        "brownian": BM
    }[config.dataset]
    
    if config.dataset == "brownian":
        training_set = dataset(
        partition="train",
        n_lags=config.n_lags,
        config=config
        )
        test_set = dataset(
            partition="test",
            n_lags=config.n_lags,
            config=config
        )
    
    else:
        training_set = dataset(
            partition="train",
            n_lags=config.n_lags,
        )
        test_set = dataset(
            partition="test",
            n_lags=config.n_lags,
        )

    training_loader = DataLoader(
        training_set,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    # config.update({"n_lags": next(iter(test_loader))[
    #               0].shape[1]}, allow_val_change=True)
    # print("data shape:", next(iter(test_loader))[0].shape)

    # if config.conditional:
    #     config.input_dim = training_loader.dataset[0][0].shape[-1] + \
    #         config.num_classes
    # else:
    #     config.input_dim = training_loader.dataset[0][0].shape[-1]
    return training_loader, test_loader


if __name__ == '__main__':
    from src.datasets.dataloader import get_dataset
    import yaml
    import ml_collections
    config_dir = 'configs/' + 'train_gan.yaml'
    with open(config_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    train_dl, test_dl = get_dataset(config=config)
