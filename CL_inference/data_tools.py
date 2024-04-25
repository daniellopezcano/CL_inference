import os
import itertools
import numpy as np
import torch
import pickle
import datetime

def load_stored_data(path_load, list_model_names):
    
    xx = []
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
            
        if ii == 0:
            xx = loaded_xx
        else:
            xx = np.concatenate((xx, loaded_xx), axis=1)
                        
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
        self, theta, xx, normalize=False, path_save_norm=None, path_load_norm=None,
        NN_augs_batch=None, add_noise_Pk=None, kmax=0.6, boxsize_cosmic_variance=1000, # Mpc/h
    ):
        
        self.theta     = theta
        self.xx        = xx
        self.NN_cosmos = self.theta.shape[0]
        self.NN_augs   = self.xx.shape[1]
        
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
        self.boxsize_cosmic_variance = boxsize_cosmic_variance
        
    def __call__(
        self,
        batch_size,
        seed="random",
        return_indexes_sampled=False,
        indexes_cosmo=None,
        indexes_augs=None,
        use_all_dataset_augs_ordered=False,
        to_torch=False,
        device="cpu"
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
        
        tmp_xx = self.xx[indexes_cosmo]
        tmp_xx_batch = []
        for ii in range(indexes_augs.shape[1]):
            tmp_xx_batch.append(tmp_xx[np.arange(batch_size), indexes_augs[:,ii]])
        xx_batch = np.transpose(np.array(tmp_xx_batch), (1,0,2))

        if self.add_noise_Pk == "cosmic_var_gauss":     
            tmp_pk = 10**xx_batch
            kmin=-2.3
            kk = np.logspace(kmin, self.kmax, num=tmp_pk.shape[-1])
            kf = 2.0 * np.pi / self.boxsize_cosmic_variance # units of boxsize_cosmic_variance in Mpc/h
            nk = 4.0 * np.pi * (kk / kf)**2
            cosmic_var_gauss_err = np.sqrt(2.0 / nk) * tmp_pk
            tmp_pk_reshaped = np.reshape(tmp_pk, ((batch_size*NN_augs_batch),)+(tmp_pk.shape[-1],))
            tmp_pk_err_reshaped = np.reshape(cosmic_var_gauss_err, ((batch_size*NN_augs_batch),)+(tmp_pk.shape[-1],))
            N_draws = 1  # this works how it works for now considering only one sample
            samples_pk = np.random.normal(
                loc=tmp_pk_reshaped[:, np.newaxis, :],
                scale=tmp_pk_err_reshaped[:, np.newaxis, :],
                size=(tmp_pk_reshaped.shape[0], N_draws, tmp_pk_reshaped.shape[-1])
            )
            samples_pk = np.reshape(samples_pk, (batch_size,) + (NN_augs_batch,) + (N_draws,) + (tmp_pk.shape[-1],)) # this is how ot should be (more general for multiple draws)
            samples_pk = np.reshape(samples_pk, (batch_size,) + (NN_augs_batch,) + (tmp_pk.shape[-1],))  # this works how it works for now considering only one sample
            xx_batch = np.log10(samples_pk)
        
        if to_torch:
            theta_batch  = torch.from_numpy(theta_batch.astype(np.float32)).to(device)
            xx_batch  = torch.from_numpy(xx_batch.astype(np.float32)).to(device)
        
        if return_indexes_sampled:
            return theta_batch, xx_batch, indexes_cosmo, indexes_augs
        else:
            return theta_batch, xx_batch
            
            
def def_data_loader(
    path_load,
    list_model_names,
    normalize=False,
    path_save_norm=None,
    path_load_norm=None,
    NN_augs_batch=None,
    add_noise_Pk=None,
    kmax=0.6,
    boxsize_cosmic_variance=1000 # Mpc/h    
):
    loaded_theta, loaded_xx = load_stored_data(path_load=path_load, list_model_names=list_model_names)
    dset = data_loader(
        loaded_theta, loaded_xx, normalize=normalize, path_save_norm=path_save_norm, path_load_norm=path_load_norm,
        NN_augs_batch=NN_augs_batch, add_noise_Pk=add_noise_Pk, kmax=kmax, boxsize_cosmic_variance=boxsize_cosmic_variance, # Mpc/h
    )
    return dset