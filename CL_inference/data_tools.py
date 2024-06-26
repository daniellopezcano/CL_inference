"""
This module contains functions for loading generated datasets and defining dataloader structures
that can be used for drawing batches during training and other utilities

Functions:
- load_stored_data: Load stored data from specified path and model names.
- draw_indexes_augs: Draw random indexes for augmentations.

Classes:
- data_loader: Data loader class for loading and processing data.
"""

import os
import itertools
import numpy as np
import torch
import pickle
import datetime
import ipdb
import logging


def load_stored_data(path_load, list_model_names, return_len_models=False, include_baryon_params=False):
    """
    Load stored data from specified path and model names.
    
    Parameters
    ----------
    path_load : str
        Path to load the data from.
    list_model_names : list
        List of model names to load.
    return_len_models : bool, optional
        Whether to return the length of models, by default False.
    include_baryon_params : bool, optional
        Whether to include baryon parameters, by default False.
    
    Returns
    -------
    tuple
        Loaded data including theta, xx, and optionally aug_params and len_models.
    """
    logging.info('Loading stored data...')
    
    xx = []
    aug_params = []
    len_models = []
    for ii, model_name in enumerate(list_model_names):
        logging.info('Loading ' + model_name + '...')
        with open(os.path.join(path_load, model_name + '_cosmos.npy'), 'rb') as ff:
            tmp_theta = np.load(ff)
            if ii == 0:
                theta = tmp_theta
            else:
                assert np.sum(tmp_theta != theta) == 0, "All theta values must coincide for the different models!"
                theta = tmp_theta
            
        with open(os.path.join(path_load, model_name + '_xx.npy'), 'rb') as ff:
            loaded_xx = np.load(ff)
        if include_baryon_params:
            with open(os.path.join(path_load, model_name + '_extended_aug_params.npy'), 'rb') as ff:
                loaded_aug_params = np.load(ff)
            
        len_models.append(loaded_xx.shape[1])
        
        if ii == 0:
            xx = loaded_xx
            if include_baryon_params:
                aug_params = loaded_aug_params
        else:
            xx = np.concatenate((xx, loaded_xx), axis=1)
            if include_baryon_params:
                aug_params = np.concatenate((aug_params, loaded_aug_params), axis=1)
        
    if include_baryon_params:
        if return_len_models:
            return theta, xx, aug_params, np.array(len_models)
        else:
            return theta, xx, aug_params
    else:
        if return_len_models:
            return theta, xx, np.array(len_models)
        else:
            return theta, xx


def draw_indexes_augs(NN_augs, batch_size, NN_aug_draws, seed=0):
    """
    Draw random indexes for augmentations.
    
    Parameters
    ----------
    NN_augs : int
        Number of augmentations.
    batch_size : int
        Batch size.
    NN_aug_draws : int
        Number of augmentation draws.
    seed : int, optional
        Random seed for reproducibility, by default 0.
    
    Returns
    -------
    numpy.ndarray
        Array of drawn indexes.
    """
    logging.debug('Drawing indexes for augmentations...')
    
    np.random.seed(seed=seed)
    all_combinations = np.array(list(itertools.combinations(np.arange(NN_augs), NN_aug_draws)))
    assert len(all_combinations) > 0, "You are asking for more augmentation draws than available"
    indexes_rand_draw_aug = np.random.choice(len(all_combinations), batch_size)
    indexes_augs = all_combinations[indexes_rand_draw_aug]
    for ii in range(indexes_augs.shape[0]):
        np.random.shuffle(indexes_augs[ii])
    return indexes_augs
    

class data_loader():
    """
    Data loader class for loading and processing data.
    
    Parameters
    ----------
    theta : numpy.ndarray
        Array of theta values.
    xx : numpy.ndarray
        Array of x values.
    aug_params : numpy.ndarray, optional
        Array of augmentation parameters, by default None.
    normalize : bool, optional
        Whether to normalize the data, by default False.
    path_save_norm : str, optional
        Path to save normalization parameters, by default None.
    path_load_norm : str, optional
        Path to load normalization parameters, by default None.
    NN_augs_batch : int, optional
        Number of augmentation draws per batch, by default None.
    add_noise_Pk : str, optional
        Method to add noise to Pk, by default None.
    kmax : float, optional
        Maximum wavenumber, by default 0.6.
    """
    
    def __init__(
        self, theta, xx, aug_params=None, normalize=False, path_save_norm=None, path_load_norm=None,
        NN_augs_batch=None, add_noise_Pk=None, kmax=0.6
    ):
        logging.info('Initializing data loader...')
        
        self.theta      = theta
        self.xx         = xx
        self.aug_params = aug_params
        self.NN_cosmos  = self.theta.shape[0]
        self.NN_augs    = self.xx.shape[1]
        
        if normalize and (path_save_norm is not None) and (path_load_norm is None):
            logging.info('Normalizing data and saving normalization parameters...')
            tmp_xx = np.reshape(self.xx, tuple([self.xx.shape[0] * self.xx.shape[1],] + list(self.xx.shape[2:])))
            tmp_mean = np.mean(tmp_xx, axis=0)
            tmp_std = np.std(tmp_xx, axis=0)
            if not os.path.exists(path_save_norm):
                os.makedirs(path_save_norm)
            np.save(os.path.join(path_save_norm, 'mean.npy'), tmp_mean)
            np.save(os.path.join(path_save_norm, 'std.npy'), tmp_std)
            self.norm_mean = tmp_mean
            self.norm_std = tmp_std
        elif normalize and (path_load_norm is not None) and (path_save_norm is None):
            logging.info('Loading normalization parameters...')
            self.norm_mean = np.load(os.path.join(path_load_norm, 'mean.npy'))
            self.norm_std = np.load(os.path.join(path_load_norm, 'std.npy'))
        else:
            self.norm_mean = 0.
            self.norm_std = 1.
        self.xx = (self.xx - self.norm_mean) / self.norm_std        
        
        self.NN_augs_batch = NN_augs_batch
        if self.NN_augs_batch is None:
            self.NN_augs_batch = self.NN_augs
        assert self.NN_augs >= self.NN_augs_batch, "You are asking for more augmentation draws than available"
        
        self.add_noise_Pk = add_noise_Pk
        self.kmax = kmax
        
    def __call__(
        self,
        batch_size,
        seed="random",
        return_indexes_sampled=False,
        indexes_cosmo=None,
        indexes_augs=None,
        use_all_dataset_augs_ordered=False,
        to_torch=False,
        device="cpu",
        box=2000,
        gaussian_error_counter_tolerance=20,
        factor_kmin_cut=4
    ):
        """
        Call function to generate data batches.
        
        Parameters
        ----------
        batch_size : int
            Batch size.
        seed : str or int, optional
            Random seed, by default "random".
        return_indexes_sampled : bool, optional
            Whether to return sampled indexes, by default False.
        indexes_cosmo : numpy.ndarray, optional
            Pre-defined cosmological indexes, by default None.
        indexes_augs : numpy.ndarray, optional
            Pre-defined augmentation indexes, by default None.
        use_all_dataset_augs_ordered : bool, optional
            Whether to use all dataset augmentations in order, by default False.
        to_torch : bool, optional
            Whether to convert data to torch tensors, by default False.
        device : str, optional
            Device for torch tensors, by default "cpu".
        box : int, optional
            Box size, by default 2000.
        gaussian_error_counter_tolerance : int, optional
            Tolerance for Gaussian error counter, by default 20.
        factor_kmin_cut : int, optional
            Factor for minimum wavenumber cut, by default 4.
        
        Returns
        -------
        tuple
            Batches of theta, xx, and optionally aug_params and sampled indexes.
        """
        logging.debug('Generating data batch...')
        
        if seed == "random":
            seed = datetime.datetime.now().microsecond % 13037
        np.random.seed(seed=seed)

        if type(indexes_cosmo) != type(np.array([])):
            indexes_cosmo = np.random.choice(self.NN_cosmos, batch_size, replace=False)
        if type(indexes_augs) != type(np.array([])):
            indexes_augs = draw_indexes_augs(self.NN_augs, batch_size, self.NN_augs_batch, seed=seed)
        if use_all_dataset_augs_ordered:
            indexes_cosmo = np.arange(self.NN_cosmos)
            indexes_augs = np.repeat(np.arange(self.NN_augs)[np.newaxis], repeats=self.NN_cosmos, axis=0)
        
        batch_size = indexes_cosmo.shape[0]
        NN_augs_batch = indexes_augs.shape[-1]
            
        theta_batch = self.theta[indexes_cosmo]
        if to_torch:
             theta_batch  = torch.from_numpy(theta_batch.astype(np.float32)).to(device)
             
        if self.aug_params is not None:
            tmp_aug_params = self.aug_params[indexes_cosmo]
            tmp_aug_params_batch = []
            for ii in range(indexes_augs.shape[1]):
                tmp_aug_params_batch.append(tmp_aug_params[np.arange(batch_size), indexes_augs[:, ii]])
            aug_params_batch = np.transpose(np.array(tmp_aug_params_batch), (1, 0, 2))
            if to_torch:
                aug_params_batch = torch.from_numpy(aug_params_batch.astype(np.float32)).to(device)
        else:
            aug_params_batch = None
            
        tmp_xx = self.xx[indexes_cosmo]
        tmp_xx_batch = []
        for ii in range(indexes_augs.shape[1]):
            tmp_xx_batch.append(tmp_xx[np.arange(batch_size), indexes_augs[:, ii]])
        xx_batch = np.transpose(np.array(tmp_xx_batch), (1, 0, 2))
        
        if self.add_noise_Pk == "cosmic_var_gauss":
            tmp_pk = 10**(xx_batch * self.norm_std + self.norm_mean)
            
            kf = 2.0 * np.pi / box
            kmin = np.log10(factor_kmin_cut * kf)
            N_kk = int((self.kmax - kmin) / (8 * kf))
            kk_log = np.logspace(kmin, self.kmax, num=N_kk)
            delta_log10kk = (np.log10(kk_log[1]) - np.log10(kk_log[0])) / 2
            kk_edges_log = 10**np.append(np.log10(kk_log) - delta_log10kk, np.log10(kk_log[-1]) + delta_log10kk)
            delta_kk = np.diff(kk_edges_log)
            cosmic_var_gauss_err = np.sqrt((4 * np.pi**2) / (box**3 * kk_log**2 * delta_kk)) * tmp_pk
            
            valid_sample = False
            while_counter = 0
            while not valid_sample:
                samples_pk = np.random.normal(loc=tmp_pk, scale=cosmic_var_gauss_err, size=tmp_pk.shape)
                if np.sum(samples_pk < 0) == 0:
                    valid_sample = True
                else:
                    while_counter += 1
                    tmp_indexes = np.where(samples_pk < 0)
                    logging.warning(f"WARNING ({while_counter} / {gaussian_error_counter_tolerance}): gaussian error approximation failed. "
                                    f"# Corrupted samples = {len(tmp_indexes[0])} / {samples_pk.shape[0] * samples_pk.shape[0]}")
                    if while_counter > gaussian_error_counter_tolerance:
                        import matplotlib as mpl
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                        ax.set_title(f'# Corrupted samples = {len(tmp_indexes[0])} / {samples_pk.shape[0] * samples_pk.shape[0]}', fontsize=16)
                        ax.set_xlabel(r'$\mathrm{Wavenumber}\, k \left[ h\, \mathrm{Mpc}^{-1} \right]$')
                        ax.set_ylabel(r'$P(k) \left[ \left(h^{-1} \mathrm{Mpc}\right)^{3} \right]$')
                        ax.set_xscale('log')
                        ax.set_yscale('log')
                        ax.plot(kk_log, samples_pk[tmp_indexes[0], tmp_indexes[1]].T, c='k', alpha=0.9, marker=None, lw=0.5, ms=2)
                        fig.tight_layout()
                        fig.savefig('./gaussian_error_approximation_failed.png')
                        assert False, f"ERROR: gaussian error approximation failed!!. Cosmology indexes: {tmp_indexes[0]}. Augmentation indexes: {tmp_indexes[1]}"
                        
            xx_batch = (np.log10(samples_pk) - self.norm_mean) / self.norm_std 
        
        if to_torch:
            xx_batch = torch.from_numpy(xx_batch.astype(np.float32)).to(device)
        
        if return_indexes_sampled:
            return theta_batch, xx_batch, aug_params_batch, indexes_cosmo, indexes_augs
        else:
            return theta_batch, xx_batch, aug_params_batch

            
def def_data_loader(
    path_load,
    list_model_names,
    normalize=False,
    path_save_norm=None,
    path_load_norm=None,
    NN_augs_batch=None,
    add_noise_Pk=None,
    kmax=0.6,
    include_baryon_params=False
):
    """
    Define data loader with specified parameters.
    
    Parameters
    ----------
    path_load : str
        Path to load data from.
    list_model_names : list
        List of model names to load.
    normalize : bool, optional
        Whether to normalize the data, by default False.
    path_save_norm : str, optional
        Path to save normalization parameters, by default None.
    path_load_norm : str, optional
        Path to load normalization parameters, by default None.
    NN_augs_batch : int, optional
        Number of augmentation draws per batch, by default None.
    add_noise_Pk : str, optional
        Method to add noise to Pk, by default None.
    kmax : float, optional
        Maximum wavenumber, by default 0.6.
    include_baryon_params : bool, optional
        Whether to include baryon parameters, by default False.
    
    Returns
    -------
    data_loader
        Instance of the data_loader class.
    """
    logging.info('Defining data loader...')
    
    if include_baryon_params:
        loaded_theta, loaded_xx, loaded_aug_params = load_stored_data(path_load=path_load, list_model_names=list_model_names, include_baryon_params=include_baryon_params)
    else:
        loaded_theta, loaded_xx = load_stored_data(path_load=path_load, list_model_names=list_model_names, include_baryon_params=include_baryon_params)
        loaded_aug_params = None
    
    dset = data_loader(
        loaded_theta, loaded_xx, aug_params=loaded_aug_params, normalize=normalize, path_save_norm=path_save_norm, path_load_norm=path_load_norm,
        NN_augs_batch=NN_augs_batch, add_noise_Pk=add_noise_Pk, kmax=kmax
    )
    
    return dset
