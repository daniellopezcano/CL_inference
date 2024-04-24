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