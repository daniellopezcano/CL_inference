"""
Perform a hyperparameter sweep using Weights & Biases (wandb) https://wandb.ai/site.

Functions:
- wandb_train               : Train a model using wandb for logging and configuration.
- wandb_sweep               : Perform a hyperparameter sweep using wandb.
- wrapper_train_from_config : Wrapper function to train a model based on provided configuration.
"""

import os
import yaml
from pathlib import Path
import wandb
import torch
import logging

from . import train_tools
from . import data_tools
from . import nn_tools


def wandb_train(config=None):
    """
    Train a model using wandb for logging and configuration.
    
    Parameters
    ----------
    config : dict, optional
        Configuration dictionary for wandb, by default None.
    """
    logging.info('Starting wandb training...')
    
    with wandb.init(config=config) as run:
        
        config = wandb.config
        
        loss = wrapper_train_from_config(run_name=run.name, **config)
        
        if "path_save" in config.keys():
            with open(os.path.join(config["path_save"], 'register_sweeps.txt'), 'a') as ff:
                ff.write(run.name + ' ' + str(loss) +'\n')
        
        wandb.run.summary["loss"] = loss # The key of this deictionary must correspond to the key specified in the wandb config file
        
        
def wandb_sweep(wandb_config_name, wandb_project_name, N_samples_hyperparmeters=5, path_to_config="../config_files/"):
    """
    Perform a hyperparameter sweep using wandb.
    
    Parameters
    ----------
    wandb_config_name : str
        Name of the wandb configuration file.
    wandb_project_name : str
        Name of the wandb project.
    N_samples_hyperparmeters : int, optional
        Number of hyperparameter samples, by default 5.
    path_to_config : str, optional
        Path to the configuration files, by default "../config_files/".
    """
    logging.info('Starting wandb sweep...')
    
    sweep_config = train_tools.load_config_file(
        path_to_config=path_to_config,
        config_file_name=wandb_config_name
    )
    
    wandb.login()
    
    wandb.agent(
        wandb.sweep(
            sweep_config,
            project=wandb_project_name
        ),
        wandb_train,
        count=N_samples_hyperparmeters
    )


def wrapper_train_from_config(
    path_save,
    path_load,
    list_model_names,
    normalize,
    NN_augs_batch,
    add_noise_Pk,
    box,
    factor_kmin_cut,
    kmax,
    include_baryon_params,
    train_mode,
    inference_loss,
    load_encoder_model_path,
    input_encoder,
    hidden_layers_encoder,
    output_encoder,
    hidden_layers_projector,
    output_projector,
    hidden_layers_inference,
    NN_params_out,
    NN_epochs,
    NN_batches_per_epoch,
    batch_size,
    lr,
    weight_decay,
    clip_grad_norm,
    seed_mode,
    seed,
    CL_loss,
    loss_hyperparameters,
    run_name,
    N_threads=1,
    load_projector_model_path=None,
    load_inference_model_path=None,
):
    """
    Wrapper function to train a model based on provided configuration.
    
    Parameters
    ----------
    (numerous parameters related to model configuration and training, see original function definition)
    
    Returns
    -------
    float
        Minimum validation loss.
    """
    logging.info('Starting training with configuration...')
    
    if run_name:
        path_save = os.path.join(path_save, run_name)
    logging.info('path_save: %s', path_save)
    
    train_tools.set_N_threads_(N_threads=N_threads)
    device = train_tools.set_torch_device_()
    
    # ---------------------------------------------------------------------- #
    # ------------------------------ DATASETS ------------------------------ #
    # ---------------------------------------------------------------------- #

    dsets = {}

    dset_name = "TRAIN"
    dsets[dset_name] = data_tools.def_data_loader(
        path_load               = os.path.join(path_load, dset_name),
        list_model_names        = list_model_names,
        normalize               = normalize,
        path_save_norm          = path_save,
        path_load_norm          = None,
        NN_augs_batch           = NN_augs_batch,
        add_noise_Pk            = add_noise_Pk,
        kmax                    = kmax,
        include_baryon_params   = include_baryon_params
    )

    dset_name = "VAL"
    dsets[dset_name] = data_tools.def_data_loader(
        path_load               = os.path.join(path_load, dset_name),
        list_model_names        = list_model_names,
        normalize               = normalize,
        path_save_norm          = None,
        path_load_norm          = path_save,
        NN_augs_batch           = NN_augs_batch,
        add_noise_Pk            = add_noise_Pk,
        kmax                    = kmax,
        include_baryon_params   = include_baryon_params
    )

    # ---------------------------------------------------------------------- #
    # -------------------------------- MODELS ------------------------------ #
    # ---------------------------------------------------------------------- #

    # ----------------------- define model encoder ----------------------- #

    assert input_encoder == dsets["TRAIN"].xx.shape[-1], "input_encoder from config file must coincide with xx size"

    if train_mode == "train_inference_fully_supervised":
        model_encoder = nn_tools.define_MLP_model(
            hidden_layers_encoder+[output_encoder], input_encoder, bn=True, last_bias=True
        ).to(device)
    else:
        model_encoder = nn_tools.define_MLP_model(
            hidden_layers_encoder+[output_encoder], input_encoder, bn=True
        ).to(device)
    if load_encoder_model_path != 'None':
        model_encoder.load_state_dict(torch.load(load_encoder_model_path))
        model_encoder.eval();

    # ----------------------- define model projector ----------------------- #

    if len(hidden_layers_projector) != 0:
        model_projector = nn_tools.define_MLP_model(
            hidden_layers_projector+[output_projector], output_encoder, bn=True
        ).to(device)
        if load_projector_model_path != 'None':
            model_projector.load_state_dict(torch.load(load_projector_model_path))
            model_projector.eval();
    else:
        model_projector=None

    # ----------------------- define model inference ----------------------- #

    if len(hidden_layers_inference) != 0:
        if inference_loss == "MSE":
            output_dim_inference = NN_params_out
        else:
            n_tril = int(NN_params_out * (NN_params_out + 1) / 2)  # Number of parameters in lower triangular matrix, for symmetric matrix
            output_dim_inference = NN_params_out + n_tril  # Dummy output of neural network

        model_inference = nn_tools.define_MLP_model(
            hidden_layers_inference+[output_dim_inference], output_encoder, bn=True
        ).to(device)        
        if load_inference_model_path != 'None':
            model_inference.load_state_dict(torch.load(load_inference_model_path))
            model_inference.eval();
    else:
        model_inference = None

    # ---------------------------------------------------------------------- #
    # -------------------------------- TRAIN ------------------------------- #
    # ---------------------------------------------------------------------- #

    dict_loss = dict(CL_loss = CL_loss, loss_hyperparameters = loss_hyperparameters, inference_loss = inference_loss)
    kwargs = dict(train=dict_loss,  val=dict_loss)

    min_val_loss = train_tools.train_model(
        dset_train=dsets["TRAIN"],
        train_mode=train_mode,
        model_encoder=model_encoder, model_projector=model_projector, model_inference=model_inference,
        NN_epochs=NN_epochs, NN_batches_per_epoch=NN_batches_per_epoch, lr=lr, weight_decay=weight_decay, clip_grad_norm=clip_grad_norm,
        batch_size=batch_size,
        dset_val=dsets["VAL"], batch_size_val=int(dsets["VAL"].theta.shape[0]/6),
        seed_mode=seed_mode, # 'random', 'deterministic' or 'overfit'
        seed=seed, # only relevant if mode is 'overfit'
        path_save=path_save,
        box=box,
        factor_kmin_cut=factor_kmin_cut,
        **kwargs
    )
    
    return min_val_loss
