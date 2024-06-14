import os
import numpy as np
import scipy as sp

def sample_latin_hypercube(dict_bounds, N_points=3000, seed=0):
    
    sample = sp.stats.qmc.LatinHypercube(d=len(dict_bounds), seed=seed).random(n=N_points)
    
    l_bounds = []
    u_bounds = []
    for key in dict_bounds.keys():
        l_bounds.append(dict_bounds[key][0])
        u_bounds.append(dict_bounds[key][1])
    
    theta_latin_hypercube = sp.stats.qmc.scale(sample, l_bounds, u_bounds)
    
    return theta_latin_hypercube


def bacco_emulator(baccoemu_input, kmax=0.6, mode=None, return_kk=False, box=2000): # bacco range: [kmin=-2.3, kmax=0.6]
    
    import baccoemu
    
    kf = 2.0 * np.pi / box
    kmin=np.log10(4*kf)
    N_kk = int((kmax - kmin) / (8*kf))
        
    if mode == None:
        mode='baryons'
    
    emulator = baccoemu.Matter_powerspectrum()
    
    kk = np.logspace(kmin, kmax, num=N_kk)
    if mode == 'linear':
        kk, pk = emulator.get_linear_pk(k=kk, cold=False, baryonic_boost=False, **baccoemu_input)
    if mode == 'nonlinear':
        kk, pk = emulator.get_nonlinear_pk(k=kk, cold=False, baryonic_boost=False, **baccoemu_input)
    if mode == 'baryons':
        kk, pk = emulator.get_nonlinear_pk(k=kk, cold=False, baryonic_boost=True, **baccoemu_input)
    
    if return_kk:
        xx = np.log10(pk)
        return kk, xx
    else:
        xx = np.log10(pk)
        return xx


def generate_baccoemu_dataset(
    NN_samples_cosmo,
    NN_samples_augs,
    dict_bounds_cosmo = dict(
        omega_cold    = [0.23, 0.40],
        omega_baryon  = [0.04, 0.06],
        hubble        = [0.60, 0.80],
        ns            = [0.92, 1.01],
        sigma8_cold   = [0.73, 0.90],
        expfactor     = 1.,
        neutrino_mass = 0.,
        w0            = -1.,
        wa            = 0.
    ),
    dict_bounds_augs = dict(
        M_c       = [9.0, 15.0],
        eta       = [-0.69, 0.69],
        beta      = [-1.00, 0.69],
        M1_z0_cen = [9.0, 13.0],
        theta_out = [0., 0.47],
        theta_inn = [-2.0, -0.523],
        M_inn     = [9.0, 13.5]
    ),
    seed = 0,
    path_save = None,
    model_name = "ModelA",
    mode_baccoemu = None,
    kmax=0.6,
    box=2000
):
    
    baccoemu_input = {}

    # ------------------------ sample cosmological parameter space ------------------------ #
    
    dict_bounds_sweep = {}
    for ii, key in enumerate(dict_bounds_cosmo.keys()):
        if type(dict_bounds_cosmo[key]) == type([]):
            assert len(dict_bounds_cosmo[key]) == 2, "Please provide bounds in the format 'param_name = [min, max]'"
            dict_bounds_sweep[key] = dict_bounds_cosmo[key]
        else:
            baccoemu_input[key] = np.repeat(dict_bounds_cosmo[key], NN_samples_cosmo*NN_samples_augs)
    
    if len(dict_bounds_sweep.keys()) != 0:
        cosmos = sample_latin_hypercube(dict_bounds_sweep, N_points=NN_samples_cosmo, seed=seed)
    else:
        cosmos = np.zeros((NN_samples_cosmo, len(dict_bounds_cosmo.keys())))
        for ii, key in enumerate(dict_bounds_cosmo.keys()):
            cosmos[:,ii] = np.repeat(dict_bounds_cosmo[key], NN_samples_cosmo)
    
    for ii, key in enumerate(dict_bounds_sweep):
        baccoemu_input[key] = np.repeat(cosmos[:,ii], NN_samples_augs)

    # ------------------------ sample augmentation parameter space ------------------------ #
    
    dict_bounds_sweep = {}
    for ii, key in enumerate(dict_bounds_augs.keys()):
        if type(dict_bounds_augs[key]) == type([]):
            assert len(dict_bounds_augs[key]) == 2, "Please provide bounds in the format 'param_name = [min, max]'"
            dict_bounds_sweep[key] = dict_bounds_augs[key]
        else:
            baccoemu_input[key] = np.repeat(dict_bounds_augs[key], NN_samples_cosmo*NN_samples_augs)
    
    if len(dict_bounds_sweep.keys()) != 0:
        aug_params = sample_latin_hypercube(dict_bounds_sweep, N_points=NN_samples_cosmo*NN_samples_augs, seed=seed)
        np.random.shuffle(aug_params)
    else:
        aug_params = np.zeros((NN_samples_cosmo*NN_samples_augs, len(dict_bounds_sweep.keys())))
        for ii, key in enumerate(dict_bounds_sweep.keys()):
            aug_params[:,ii] = dict_bounds_sweep[key]
        
    for ii, key in enumerate(dict_bounds_sweep):
        baccoemu_input[key] = aug_params[:,ii]

    aug_params = np.reshape(aug_params, (NN_samples_cosmo, NN_samples_augs, aug_params.shape[-1]))

    # ------------------------ generate observations with baccoemu ------------------------ #
    
    xx = bacco_emulator(
        baccoemu_input, kmax=kmax, return_kk=False, mode=mode_baccoemu, box=box
    )

    xx = np.reshape(xx, (NN_samples_cosmo, NN_samples_augs, xx.shape[-1]))
    
    # ------------------------ reshape baccoemu_input to store all aug baryonic parameters ------------------------ #
    
    extended_aug_param_keys = ['M_c', 'eta', 'beta', 'M1_z0_cen', 'theta_out', 'theta_inn', 'M_inn']
    extended_aug_params = np.zeros((xx.shape[0], xx.shape[1], len(extended_aug_param_keys)))
    for ii, key in enumerate(extended_aug_param_keys):
        extended_aug_params[..., ii] = np.reshape(baccoemu_input[key], (xx.shape[0], xx.shape[1]))
    
    # ------------------------ Save data ------------------------ #
    
    if path_save != None:
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        np.save(os.path.join(path_save, model_name + '_cosmos.npy'), cosmos)
        np.save(os.path.join(path_save, model_name + '_xx.npy'), xx)
        np.save(os.path.join(path_save, model_name + '_aug_params.npy'), aug_params)
        np.save(os.path.join(path_save, model_name + '_extended_aug_params.npy'), extended_aug_params)
    
    return cosmos, xx, aug_params, extended_aug_params