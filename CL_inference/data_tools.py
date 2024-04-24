import os
import itertools
import numpy as np
from datetime import datetime
import pickle

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
    
    def __init__(self, theta, xx, normalize=False, path_save_norm=None, path_load_norm=None):
        
        self.theta = theta
        self.xx = xx
        self.NN_cosmos = self.theta.shape[0]
        self.NN_augs = self.xx.shape[1]
        
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

        
    def __call__(
        self,
        batch_size,
        NN_augs_batch=None,
        seed="random",
        return_indexes_sampled=False,
        indexes_cosmo=None,
        indexes_augs=None,
        add_noise_Pk=None,
        boxsize_cosmic_variance=1000, # Mpc/h
        kmax=0.6
    ):
        
        if NN_augs_batch == None:
            NN_augs_batch = self.NN_augs
        assert self.NN_augs >= NN_augs_batch, "You are asking for more augmentation draws than available"
        
        if seed == "random":
            datetime.now().microsecond %13037
        np.random.seed(seed=seed)
        
        if type(indexes_cosmo) != type(np.array([])):
            indexes_cosmo = np.random.choice(self.NN_cosmos, batch_size, replace=False)
        if type(indexes_augs) != type(np.array([])):
            indexes_augs = draw_indexes_augs(self.NN_augs, batch_size, NN_augs_batch, seed=seed)
        
        batch_size = indexes_cosmo.shape[0]
            
        theta_batch = self.theta[indexes_cosmo]
        
        tmp_xx = self.xx[indexes_cosmo]
        tmp_xx_batch = []
        for ii in range(indexes_augs.shape[1]):
            tmp_xx_batch.append(tmp_xx[np.arange(batch_size), indexes_augs[:,ii]])
        xx_batch = np.transpose(np.array(tmp_xx_batch), (1,0,2))

        if add_noise_Pk == "cosmic_var_gauss":     
            tmp_pk = 10**xx_batch
            kmin=-2.3
            kk = np.logspace(kmin, kmax, num=tmp_pk.shape[-1])
            kf = 2.0 * np.pi / boxsize_cosmic_variance # units of boxsize_cosmic_variance in Mpc/h
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

        if return_indexes_sampled:
            return theta_batch, xx_batch, indexes_cosmo, indexes_augs
        else:
            return theta_batch, xx_batch