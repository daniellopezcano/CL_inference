import os, sys
import yaml
from pathlib import Path
import torch
import numpy as np

from . import nn_tools
from . import data_tools
from . import custom_loss_functions

def load_config_file_wandb_format(path_to_config, config_file_name):
    config_wandb = yaml.safe_load(Path(os.path.join(path_to_config, config_file_name)).read_text())
    config = {}
    for ii, key in enumerate(config_wandb.keys()):
        try:
            config[key] = config_wandb[key]['value']
        except:
            print("not value for key", key)
    return config
    
    
def reload_models(save_path, main_name, evalute_mode, configs, device):

    models_encoder = {}
    models_inference = {}

    for ii, sweep_name in enumerate(configs.keys()):
        
        config = configs[sweep_name]

        if ("only_inference" in main_name) and ("CL" in main_name):
            load_encoder_model_path = config["load_encoder_model_path"]
        else:
            load_encoder_model_path = os.path.join(save_path, main_name, sweep_name, "model_encoder.pt")

        if evalute_mode != "eval_CL":
            load_inference_model_path = os.path.join(save_path, main_name, sweep_name, "model_inference.pt")
        else:
            load_inference_model_path = None

        input_encoder = config['input_encoder']
        hidden_layers_encoder = config['hidden_layers_encoder']
        output_encoder = config['output_encoder']
        hidden_layers_inference = config['hidden_layers_inference']
        NN_params_out = config['NN_params_out']
        inference_loss = config['inference_loss']
        train_mode = config['train_mode']

        # ----------------------- define model encoder ----------------------- #
        
        if train_mode == "train_inference_fully_supervised":
            models_encoder[sweep_name] = nn_tools.define_MLP_model(
                hidden_layers_encoder+[output_encoder], input_encoder, bn=True, last_bias=True
            ).to(device)
        else:
            models_encoder[sweep_name] = nn_tools.define_MLP_model(
                hidden_layers_encoder+[output_encoder], input_encoder, bn=True
            ).to(device)

        models_encoder[sweep_name].load_state_dict(torch.load(load_encoder_model_path))
        models_encoder[sweep_name].eval();
        print("Model encoder loaded: " + load_encoder_model_path)

        # ----------------------- define model inference ----------------------- #
        
        if len(hidden_layers_inference) != 0:
            if inference_loss == "MSE":
                output_dim_inference = NN_params_out
            else:
                n_tril = int(NN_params_out * (NN_params_out + 1) / 2)  # Number of parameters in lower triangular matrix, for symmetric matrix
                output_dim_inference = NN_params_out + n_tril  # Dummy output of neural network

            models_inference[sweep_name] = nn_tools.define_MLP_model(
                hidden_layers_inference+[output_dim_inference], output_encoder, bn=True
            ).to(device)
        else:
            models_inference[sweep_name] = None
            
        if load_inference_model_path != None:
            models_inference[sweep_name] = nn_tools.define_MLP_model(hidden_layers_inference+[output_dim_inference], output_encoder, bn=True).to(device)
            models_inference[sweep_name].load_state_dict(torch.load(load_inference_model_path))
            models_inference[sweep_name].eval();
            print("Model inference loaded: " + load_inference_model_path)
    
    return models_encoder, models_inference
    
    
def compute_dataset_results(config, sweep_name, list_model_names, models_encoder, models_inference, device, dset_key="TEST", indexes_cosmo=None, use_all_dataset_augs_ordered=True):
    
    loaded_theta, loaded_xx, len_models = data_tools.load_stored_data(
        path_load=os.path.join(config['path_load'], dset_key),
        list_model_names=list_model_names,
        return_len_models=True
    )

    dset = data_tools.data_loader(
        loaded_theta,
        loaded_xx,
        normalize=config['normalize'],
        path_load_norm = os.path.join(config['path_save'], sweep_name),
        NN_augs_batch = np.sum(len_models),
        add_noise_Pk=config['add_noise_Pk'],
        kmax=config['kmax'],
        boxsize_cosmic_variance=config['boxsize_cosmic_variance'], # Mpc/h
    )
    if type(indexes_cosmo) == type(np.array([])):
        indexes_augs=np.repeat(np.arange(dset.NN_augs)[np.newaxis], repeats=len(indexes_cosmo), axis=0)
    else:
        indexes_augs=None
    theta_true, xx, indexes_cosmo, indexes_augs = dset(
        0, seed=0, to_torch=True, device=device, use_all_dataset_augs_ordered=use_all_dataset_augs_ordered, indexes_cosmo=indexes_cosmo, indexes_augs=indexes_augs, return_indexes_sampled=True
    )
    theta_true = theta_true.cpu().detach().numpy()
    
    NN_params_out = config["NN_params_out"]
    tmp_xx = torch.reshape(xx, (np.prod(xx.shape[:2]),) + (xx.shape[-1],))
    thetas_pred = np.zeros((len(models_encoder.keys()), tmp_xx.shape[0], NN_params_out))
    Covs = np.zeros((len(models_encoder.keys()), tmp_xx.shape[0], NN_params_out, NN_params_out))
    hh = {}
    for ii, sweep_name in enumerate(models_encoder.keys()):
        hh[sweep_name] = models_encoder[sweep_name](tmp_xx.contiguous())
        yy = models_inference[sweep_name](hh[sweep_name].contiguous())

        theta_pred, cov_pred = yy[:, :theta_true.shape[-1]], yy[:, theta_true.shape[-1]:]
        Cov = custom_loss_functions.vector_to_Cov(cov_pred).to(device=device)

        thetas_pred[ii] = theta_pred.cpu().detach().numpy()
        Covs[ii] = Cov.cpu().detach().numpy()
        
        hh[sweep_name] = hh[sweep_name].cpu().detach().numpy()
        hh[sweep_name] = np.reshape(hh[sweep_name], (indexes_augs.shape[0], indexes_augs.shape[1], hh[sweep_name].shape[-1]))
        
    xx = xx.cpu().detach().numpy()
        
    theta_pred = np.mean(thetas_pred, axis=0)
    Cov = np.mean(Covs, axis=0)
    
    theta_pred = np.reshape(theta_pred, (indexes_augs.shape[0], dset.NN_augs_batch, dset.theta.shape[-1]))
    Cov = np.reshape(Cov, (indexes_augs.shape[0], indexes_augs.shape[1], dset.theta.shape[-1], dset.theta.shape[-1]))
    
    return xx, hh, theta_true, theta_pred, Cov, len_models