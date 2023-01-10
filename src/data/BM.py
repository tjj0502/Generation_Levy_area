import numpy as np
import torch
from torch import nn
import signatory
import pathlib
import os
from src.utils import train_test_split


def get_BM_paths(dim=2, size=20000, n_lags=50, T=0.1):
    r"""
    Paths of Brownian Motion up to time T 
    
    Parameters
    ----------
    dim: int
    size: int
        size of the dataset
    n_lags: int
        Number of timesteps in the path

    Returns
    -------
    dataset: torch.tensor
        array of shape (size, n_lags, dim)

    """

    print('Generate 20000 path samples of Brownian motion')
    BM_paths = torch.empty(size, n_lags, dim)
    nn.init.normal_(BM_paths, 0, np.sqrt(T / n_lags))
    BM_paths[:, 0, :] *= 0
    BM_paths = torch.cumsum(BM_paths, dim=1)
    return BM_paths



class BM(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: str,
        n_lags: int,
        config,
        **kwargs,
    ):
        n_lags = n_lags

        data_loc = pathlib.Path(
            config.exp_dir + '/processed_data_{}'.format(n_lags))

        if os.path.exists(data_loc):
            pass
        else:
            if not os.path.exists(data_loc.parent):
                os.mkdir(data_loc.parent)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            x_real = get_BM_paths(n_lags=n_lags, T = config.T)
            x_real = signatory.logsignature(x_real, config.logsig_level, mode='brackets')
            train_X, test_X = train_test_split(x_real, 0.8)
            save_data(
                data_loc,
                train_X=train_X,
                test_X=test_X,
            )
        X = self.load_data(data_loc, partition)
        super(BM, self).__init__(X)

    @staticmethod
    def load_data(data_loc, partition):

        tensors = load_data(data_loc)
        if partition == "train":
            X = tensors["train_X"]
        elif partition == "test":
            X = tensors["test_X"]
        else:
            raise NotImplementedError(
                "the set {} is not implemented.".format(set))

        return X


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + ".pt")


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith(".pt"):
            tensor_name = filename.split(".")[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors
