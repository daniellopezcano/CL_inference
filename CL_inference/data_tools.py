import os
import itertools
import numpy as np
import torch
import pickle
import datetime

def load_stored_data(path_load, list_model_names, return_len_models=False, include_baryon_params=False):
    
    xx = []
    aug_params = []
    len_models = []
    for ii, model_name in enumerate(list_model_names):
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
    np.random.seed(seed=seed)
    all_combinations = np.array(list(itertools.combinations(np.arange(NN_augs), NN_aug_draws)))
    assert len(all_combinations) > 0, "You are asking for more augmentation draws than available"
    indexes_rand_draw_aug = np.random.choice(len(all_combinations), batch_size)
    indexes_augs = all_combinations[indexes_rand_draw_aug]
    for ii in range(indexes_augs.shape[0]):
        np.random.shuffle(indexes_augs[ii])
    return indexes_augs
    

class data_loader():
    
    def __init__(
        self, theta, xx, aug_params=None, normalize=False, path_save_norm=None, path_load_norm=None,
        NN_augs_batch=None, add_noise_Pk=None, kmax=0.6
    ):
        
        self.theta      = theta
        self.xx         = xx
        self.aug_params = aug_params
        self.NN_cosmos  = self.theta.shape[0]
        self.NN_augs    = self.xx.shape[1]
        
        if normalize and (path_save_norm != None) and (path_load_norm == None):
            tmp_xx = np.reshape(self.xx, tuple([self.xx.shape[0]*self.xx.shape[1],] + list(self.xx.shape[2:])))
            tmp_mean = np.mean(tmp_xx, axis=0)
            tmp_std = np.std(tmp_xx, axis=0)
            if not os.path.exists(path_save_norm):
                os.makedirs(path_save_norm)
            np.save(os.path.join(path_save_norm, 'mean.npy'), tmp_mean)
            np.save(os.path.join(path_save_norm, 'std.npy'), tmp_std)
            self.norm_mean = tmp_mean
            self.norm_std = tmp_std
        elif normalize and (path_load_norm != None) and (path_save_norm == None):
            self.norm_mean = np.load(os.path.join(path_load_norm, 'mean.npy'))
            self.norm_std = np.load(os.path.join(path_load_norm, 'std.npy'))
        else:
            self.norm_mean = 0.
            self.norm_std = 1.
        self.xx = (self.xx - self.norm_mean) / self.norm_std        
        
        self.NN_augs_batch = NN_augs_batch
        if self.NN_augs_batch == None:
            self.NN_augs_batch = self.NN_augs
        assert self.NN_augs >= self.NN_augs_batch, "You are asking for more augmentation draws than available"
        
        self.add_noise_Pk            = add_noise_Pk
        self.kmax                    = kmax
        
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
        box=2000
    ):
        
        if seed == "random":
            datetime.datetime.now().microsecond %13037
        np.random.seed(seed=seed)

        if type(indexes_cosmo) != type(np.array([])):
            indexes_cosmo = np.random.choice(self.NN_cosmos, batch_size, replace=False)
        if type(indexes_augs) != type(np.array([])):
            indexes_augs = draw_indexes_augs(self.NN_augs, batch_size, self.NN_augs_batch, seed=seed)
        if use_all_dataset_augs_ordered:
            indexes_cosmo=np.arange(self.NN_cosmos)
            indexes_augs=np.repeat(np.arange(self.NN_augs)[np.newaxis], repeats=self.NN_cosmos, axis=0)
            
        batch_size = indexes_cosmo.shape[0]
        NN_augs_batch = indexes_augs.shape[-1]
            
        theta_batch = self.theta[indexes_cosmo]
        if to_torch:
             theta_batch  = torch.from_numpy(theta_batch.astype(np.float32)).to(device)
             
        if self.aug_params is not None:
            tmp_aug_params = self.aug_params[indexes_cosmo]
            tmp_aug_params_batch = []
            for ii in range(indexes_augs.shape[1]):
                tmp_aug_params_batch.append(tmp_aug_params[np.arange(batch_size), indexes_augs[:,ii]])
            aug_params_batch = np.transpose(np.array(tmp_aug_params_batch), (1,0,2))
            if to_torch:
                aug_params_batch = torch.from_numpy(aug_params_batch.astype(np.float32)).to(device)
        else:
            aug_params_batch = None
            
        tmp_xx = self.xx[indexes_cosmo]
        tmp_xx_batch = []
        for ii in range(indexes_augs.shape[1]):
            tmp_xx_batch.append(tmp_xx[np.arange(batch_size), indexes_augs[:,ii]])
        xx_batch = np.transpose(np.array(tmp_xx_batch), (1,0,2))
        
        if self.add_noise_Pk == "cosmic_var_gauss":
            tmp_pk = 10**(xx_batch*self.norm_std + self.norm_mean)
            
            kf = 2.0 * np.pi / box
            kmin=np.log10(4*kf)
            N_kk = int((self.kmax - kmin) / (8*kf))
            kk_log = np.logspace(kmin, self.kmax, num=N_kk)
            delta_log10kk = (np.log10(kk_log[1]) - np.log10(kk_log[0])) / 2
            kk_edges_log = 10**np.append(np.log10(kk_log)-delta_log10kk, np.log10(kk_log[-1])+delta_log10kk)
            delta_kk = np.diff(kk_edges_log)
            cosmic_var_gauss_err = np.sqrt( (4*np.pi**2) / (box**3 * kk_log**2 * delta_kk) ) * tmp_pk
            
            valid_sample = False
            while_counter = 0
            while not valid_sample:
                samples_pk = np.random.normal(
                    loc=tmp_pk,
                    scale=cosmic_var_gauss_err,
                    size=tmp_pk.shape
                )
                if np.sum(samples_pk<0) == 0:
                    valid_sample = True
                else:
                    print("WARNING: gaussian errror approximation failed")
                    assert while_counter < 10, "ERROR: gaussian errror approximation failed!!"
                    while_counter += 1
                    # import matplotlib as mpl
                    # import matplotlib.pyplot as plt
                    # fig, ax = mpl.pyplot.subplots(1,1,figsize=(6,4))
                    # ax.set_xlabel(r'$\mathrm{Wavenumber}\, k \left[ h\, \mathrm{Mpc}^{-1} \right]$')
                    # ax.set_ylabel(r'$P(k) \left[ \left(h^{-1} \mathrm{Mpc}\right)^{3} \right]$')
                    # ax.set_xscale('log')
                    # ax.set_yscale('log')
                    # ax.plot(kk_log, samples_pk[:,0].T, c='limegreen', alpha=0.8, marker='o', ms=2)
                    # fig.set_tight_layout(True)
                    # fig.savefig("/cosmos_storage/home/dlopez/Projects/CL_inference/models/" + str(seed) + 'eval_loss.png')
            xx_batch = (np.log10(samples_pk) - self.norm_mean) / self.norm_std 
        
        if to_torch:
            xx_batch  = torch.from_numpy(xx_batch.astype(np.float32)).to(device)
        
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