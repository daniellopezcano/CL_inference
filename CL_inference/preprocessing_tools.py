"""
This module contains functions for generating power spectrum datasets
employing baccoemu https://baccoemu.readthedocs.io/en/latest/

Functions:
- sample_latin_hypercube: Generate samples using Latin Hypercube Sampling (LHS) within given bounds.
- bacco_emulator: Generate power spectrum using the baccoemu emulator.
- generate_baccoemu_dataset: Generate a dataset using baccoemu emulator with specified cosmological and augmentation parameters.
"""

import os
import numpy as np
import scipy as sp
import logging


def sample_latin_hypercube(dict_bounds, N_points=3000, seed=0):
    """
    Generate samples using Latin Hypercube Sampling (LHS) within given bounds.
    
    Parameters
    ----------
    dict_bounds : dict
        Dictionary containing the bounds for each parameter.
    N_points : int, optional
        Number of sample points to generate, by default 3000.
    seed : int, optional
        Random seed for reproducibility, by default 0.
    
    Returns
    -------
    numpy.ndarray
        Array of sampled points.
    """
    logging.info('Generating Latin Hypercube samples...')
    sample = sp.stats.qmc.LatinHypercube(d=len(dict_bounds), seed=seed).random(n=N_points)
    
    l_bounds = []
    u_bounds = []
    for key in dict_bounds.keys():
        l_bounds.append(dict_bounds[key][0])
        u_bounds.append(dict_bounds[key][1])
    
    theta_latin_hypercube = sp.stats.qmc.scale(sample, l_bounds, u_bounds)
    
    logging.debug('Sampled points: %s', theta_latin_hypercube.shape)
    return theta_latin_hypercube


def bacco_emulator(baccoemu_input, kmax=0.6, mode=None, return_kk=False, box=2000, factor_kmin_cut=4):
    """
    Generate power spectrum using the baccoemu emulator.
    
    Parameters
    ----------
    baccoemu_input : dict
        Input parameters for the emulator.
    kmax : float, optional
        Maximum wavenumber, by default 0.6.
    mode : str, optional
        Mode for the emulator ('linear', 'nonlinear', 'baryons'), by default None.
    return_kk : bool, optional
        Whether to return wavenumbers, by default False.
    box : int, optional
        Box size, by default 2000.
    factor_kmin_cut : int, optional
        Factor for minimum wavenumber cut, by default 4.
    
    Returns
    -------
    numpy.ndarray
        Power spectrum or (wavenumbers, power spectrum) if return_kk is True.
    """
    logging.info('Generating power spectrum using baccoemu emulator...')
    
    try:
        import baccoemu
    except ImportError as e:
        logging.error('Failed to import baccoemu: %s', e)
        raise

    kf = 2.0 * np.pi / box
    kmin = np.log10(factor_kmin_cut * kf)
    N_kk = int((kmax - kmin) / (8 * kf))
        
    if mode is None:
        mode = 'baryons'
    
    emulator = baccoemu.Matter_powerspectrum()
    
    kk = np.logspace(kmin, kmax, num=N_kk)
    if mode == 'linear':
        kk, pk = emulator.get_linear_pk(k=kk, cold=False, baryonic_boost=False, **baccoemu_input)
    elif mode == 'nonlinear':
        kk, pk = emulator.get_nonlinear_pk(k=kk, cold=False, baryonic_boost=False, **baccoemu_input)
    elif mode == 'baryons':
        kk, pk = emulator.get_nonlinear_pk(k=kk, cold=False, baryonic_boost=True, **baccoemu_input)
    else:
        logging.warning('Unknown mode: %s. Defaulting to baryons.', mode)
        kk, pk = emulator.get_nonlinear_pk(k=kk, cold=False, baryonic_boost=True, **baccoemu_input)
    
    xx = np.log10(pk)
    logging.debug('Power spectrum: %s', xx.shape)
    
    if return_kk:
        return kk, xx
    else:
        return xx


def generate_baccoemu_dataset(
    NN_samples_cosmo,
    NN_samples_augs,
    dict_bounds_cosmo=None,
    dict_bounds_augs=None,
    seed=0,
    path_save=None,
    model_name="ModelA",
    mode_baccoemu=None,
    kmax=0.6,
    box=2000, 
    factor_kmin_cut=4
):
    """
    Generate a dataset using baccoemu emulator with specified cosmological and augmentation parameters.
    
    Parameters
    ----------
    NN_samples_cosmo : int
        Number of cosmological samples.
    NN_samples_augs : int
        Number of augmentation samples.
    dict_bounds_cosmo : dict, optional
        Bounds for cosmological parameters, by default None.
    dict_bounds_augs : dict, optional
        Bounds for augmentation parameters, by default None.
    seed : int, optional
        Random seed for reproducibility, by default 0.
    path_save : str, optional
        Path to save the generated dataset, by default None.
    model_name : str, optional
        Model name for saved files, by default "ModelA".
    mode_baccoemu : str, optional
        Mode for the baccoemu emulator, by default None.
    kmax : float, optional
        Maximum wavenumber, by default 0.6.
    box : int, optional
        Box size, by default 2000.
    factor_kmin_cut : int, optional
        Factor for minimum wavenumber cut, by default 4.
    
    Returns
    -------
    tuple
        Tuple containing the cosmological samples, power spectrum, augmentation parameters, and extended augmentation parameters.
    """
    logging.info('Generating baccoemu dataset...')
    
    if dict_bounds_cosmo is None:
        dict_bounds_cosmo = {
            'omega_cold': [0.23, 0.40],
            'omega_baryon': [0.04, 0.06],
            'hubble': [0.60, 0.80],
            'ns': [0.92, 1.01],
            'sigma8_cold': [0.73, 0.90],
            'expfactor': 1.,
            'neutrino_mass': 0.,
            'w0': -1.,
            'wa': 0.
        }
        
    if dict_bounds_augs is None:
        dict_bounds_augs = {
            'M_c': [9.0, 15.0],
            'eta': [-0.69, 0.69],
            'beta': [-1.00, 0.69],
            'M1_z0_cen': [9.0, 13.0],
            'theta_out': [0., 0.47],
            'theta_inn': [-2.0, -0.523],
            'M_inn': [9.0, 13.5]
        }
    
    baccoemu_input = {}

    # Sample cosmological parameter space
    dict_bounds_sweep = {}
    for key in dict_bounds_cosmo.keys():
        if isinstance(dict_bounds_cosmo[key], list):
            assert len(dict_bounds_cosmo[key]) == 2, "Please provide bounds in the format 'param_name = [min, max]'"
            dict_bounds_sweep[key] = dict_bounds_cosmo[key]
        else:
            baccoemu_input[key] = np.repeat(dict_bounds_cosmo[key], NN_samples_cosmo * NN_samples_augs)
    
    if dict_bounds_sweep:
        cosmos = sample_latin_hypercube(dict_bounds_sweep, N_points=NN_samples_cosmo, seed=seed)
    else:
        cosmos = np.zeros((NN_samples_cosmo, len(dict_bounds_cosmo.keys())))
        for ii, key in enumerate(dict_bounds_cosmo.keys()):
            cosmos[:, ii] = np.repeat(dict_bounds_cosmo[key], NN_samples_cosmo)
    
    for ii, key in enumerate(dict_bounds_sweep):
        baccoemu_input[key] = np.repeat(cosmos[:, ii], NN_samples_augs)

    # Sample augmentation parameter space
    dict_bounds_sweep = {}
    for key in dict_bounds_augs.keys():
        if isinstance(dict_bounds_augs[key], list):
            assert len(dict_bounds_augs[key]) == 2, "Please provide bounds in the format 'param_name = [min, max]'"
            dict_bounds_sweep[key] = dict_bounds_augs[key]
        else:
            baccoemu_input[key] = np.repeat(dict_bounds_augs[key], NN_samples_cosmo * NN_samples_augs)
    
    if dict_bounds_sweep:
        aug_params = sample_latin_hypercube(dict_bounds_sweep, N_points=NN_samples_cosmo * NN_samples_augs, seed=seed)
        np.random.shuffle(aug_params)
    else:
        aug_params = np.zeros((NN_samples_cosmo * NN_samples_augs, len(dict_bounds_sweep.keys())))
        for ii, key in enumerate(dict_bounds_sweep.keys()):
            aug_params[:, ii] = dict_bounds_sweep[key]
        
    for ii, key in enumerate(dict_bounds_sweep):
        baccoemu_input[key] = aug_params[:, ii]

    aug_params = np.reshape(aug_params, (NN_samples_cosmo, NN_samples_augs, aug_params.shape[-1]))

    # Generate observations with baccoemu
    xx = bacco_emulator(
        baccoemu_input, kmax=kmax, return_kk=False, mode=mode_baccoemu, box=box, factor_kmin_cut=factor_kmin_cut
    )

    xx = np.reshape(xx, (NN_samples_cosmo, NN_samples_augs, xx.shape[-1]))
    
    # Reshape baccoemu_input to store all aug baryonic parameters
    extended_aug_param_keys = ['M_c', 'eta', 'beta', 'M1_z0_cen', 'theta_out', 'theta_inn', 'M_inn']
    extended_aug_params = np.zeros((xx.shape[0], xx.shape[1], len(extended_aug_param_keys)))
    for ii, key in enumerate(extended_aug_param_keys):
        extended_aug_params[..., ii] = np.reshape(baccoemu_input[key], (xx.shape[0], xx.shape[1]))
    
    # Save data
    if path_save is not None:
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        np.save(os.path.join(path_save, model_name + '_cosmos.npy'), cosmos)
        np.save(os.path.join(path_save, model_name + '_xx.npy'), xx)
        np.save(os.path.join(path_save, model_name + '_aug_params.npy'), aug_params)
        np.save(os.path.join(path_save, model_name + '_extended_aug_params.npy'), extended_aug_params)
    
    return cosmos, xx, aug_params, extended_aug_params
