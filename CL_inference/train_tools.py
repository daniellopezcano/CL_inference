"""
This module contains functions for setting up the PyTorch device, configuring thread usage, 
loading configuration files, and training machine learning models with various modes.

Functions:
- set_torch_device_: Configures and returns the PyTorch device.
- set_N_threads_: Sets the number of threads for various libraries.
- load_config_file: Loads and returns a YAML configuration file.
- train_model: Trains a model based on the specified training mode.
- train_single_epoch: Trains a model for a single epoch.
- eval_single_epoch: Evaluates a model after training for one epoch.
- eval_dataset: Evaluates a dataset and returns the computed loss.
"""

import os
import torch
import yaml
from pathlib import Path
import datetime
import ipdb
import logging

from . import custom_loss_functions


def set_torch_device_(device=None):
    """
    Sets the PyTorch device to either CPU or GPU.

    Parameters:
    - device (torch.device or str, optional): The device to set. Defaults to CUDA if available, otherwise CPU.

    Returns:
    - device (torch.device): The configured device.
    """
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f'Device: {device}')
    if device == "cuda":
        logging.info(f"GPU model: {torch.cuda.get_device_name(0)}")
    
    if device == "cpu":
        torch.manual_seed(seed=0)
    if device == "cuda":
        torch.cuda.manual_seed(seed=0)
        torch.cuda.manual_seed_all(seed=0)
    
    return device


def set_N_threads_(N_threads=1):
    """
    Sets the number of threads for various numerical libraries.

    Parameters:
    - N_threads (int): The number of threads to set.

    Returns:
    - N_threads (int): The configured number of threads.
    """
    logging.info(f'N_threads: {N_threads}')
    
    os.environ["OMP_NUM_THREADS"] = str(N_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(N_threads)
    os.environ["MKL_NUM_THREADS"] = str(N_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(N_threads)
    
    return N_threads
    
    
def load_config_file(path_to_config, config_file_name):
    """
    Loads a YAML configuration file.

    Parameters:
    - path_to_config (str): The path to the configuration directory.
    - config_file_name (str): The name of the configuration file.

    Returns:
    - config (dict): The loaded configuration.
    """
    config_path = Path(os.path.join(path_to_config, config_file_name))
    config = yaml.safe_load(config_path.read_text())
    return config


def train_model(
    dset_train,
    train_mode, # "train_CL", "train_inference_from_latents", "train_inference_fully_supervised", or "train_CL_and_inference"
    model_encoder, model_projector=None, model_inference=None,
    NN_epochs=300, NN_batches_per_epoch=10, batch_size=16, lr=1e-3, weight_decay=0., clip_grad_norm=None,
    seed_mode="random", # 'random', 'deterministic' or 'overfit'
    seed=0, # only relevant if mode is 'overfit'
    dset_val=None, batch_size_val=16, path_save=None, box=2000, factor_kmin_cut=4,
    **kwargs
    ):
    """
    Trains a model based on the specified training mode.

    Parameters:
    - dset_train: Training dataset.
    - train_mode (str): Mode of training. Must be one of 'train_CL', 'train_inference_from_latents', 'train_inference_fully_supervised', or 'train_CL_and_inference'.
    - model_encoder (torch.nn.Module): Encoder model.
    - model_projector (torch.nn.Module, optional): Projector model.
    - model_inference (torch.nn.Module, optional): Inference model.
    - NN_epochs (int): Number of training epochs.
    - NN_batches_per_epoch (int): Number of batches per epoch.
    - batch_size (int): Batch size.
    - lr (float): Learning rate.
    - weight_decay (float): Weight decay.
    - clip_grad_norm (float, optional): Gradient clipping norm.
    - seed_mode (str): Seed mode. Must be 'random', 'deterministic', or 'overfit'.
    - seed (int): Seed value (relevant if seed_mode is 'overfit').
    - dset_val: Validation dataset.
    - batch_size_val (int): Batch size for validation.
    - path_save (str, optional): Path to save the model.
    - box (float): Box size parameter.
    - factor_kmin_cut (float): Factor kmin cut parameter.
    - **kwargs: Additional keyword arguments to use internally in the functions train_single_epoch & eval_single_epoch

    Returns:
    - min_val_loss (float): Minimum validation loss.
    """
    assert train_mode in ["train_CL", "train_inference_from_latents", "train_inference_fully_supervised", "train_CL_and_inference"], "mode must belong to one of the following categories: 'train_CL', 'train_inference_from_latents', 'train_inference_fully_supervised', or 'train_CL_and_inference'"
    
    if train_mode == "train_CL":
        assert model_projector!=None, "model_projector must be provided"
        optimizer = torch.optim.AdamW([*model_encoder.parameters(), *model_projector.parameters()], lr=lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, threshold=0.01, threshold_mode='abs', factor=0.3, min_lr=1e-8, verbose=True)
    if train_mode == "train_inference_from_latents":
        assert model_inference!=None, "model_inference must be provided"
        optimizer = torch.optim.AdamW([*model_inference.parameters()], lr=lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=30, threshold=0.15, threshold_mode='abs', factor=0.3, min_lr=1e-8, verbose=True)
    if train_mode == "train_inference_fully_supervised":
        assert model_inference!=None, "model_inference must be provided"
        optimizer = torch.optim.AdamW([*model_encoder.parameters(), *model_inference.parameters()], lr=lr, betas=(0.9, 0.999), eps=1e-6, amsgrad=False, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=30, threshold=0.15, threshold_mode='abs', factor=0.3, min_lr=1e-8, verbose=True)
    if train_mode == "train_CL_and_inference":
        assert model_projector!=None, "model_projector must be provided"
        assert model_inference!=None, "model_inference must be provided"
        optimizer = torch.optim.AdamW([*model_encoder.parameters(), *model_encoder.parameters(), *model_inference.parameters()], lr=lr, betas=(0.9, 0.999), eps=1e-6, amsgrad=False, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=30, threshold=0.15, threshold_mode='abs', factor=0.3, min_lr=1e-8, verbose=True)

    assert seed_mode in ["random", "deterministic", "overfit"], " must belong to one of the following categories: 'random', 'deterministic' or 'overfit'"
    
    if next(model_encoder.parameters()).is_cuda: device = "cuda"
    else: device = "cpu"
    
    min_val_loss = None
    for tt in range(NN_epochs):
        logging.info(f"\n-------------------------------------\n-------------- Epoch {tt+1} --------------\n-------------------------------------\n")

        if seed_mode == "deterministic": seed0 = NN_batches_per_epoch*tt
        else: seed0=0
        
        # train the model one epoch
        train_single_epoch(
            dset_train=dset_train, train_mode=train_mode, optimizer=optimizer,
            model_encoder=model_encoder, model_projector=model_projector, model_inference=model_inference,
            NN_batches_per_epoch=NN_batches_per_epoch, batch_size=batch_size, clip_grad_norm=clip_grad_norm,
            seed_mode=seed_mode, seed=seed, seed0=seed0,
            device=device, save_aux_fig_name_epoch=str(tt), box=box, factor_kmin_cut=factor_kmin_cut,
            **kwargs['train']
        )
        # evaluation of the model after training one epoch
        min_val_loss, train_loss, val_loss = eval_single_epoch(
            dset_train=dset_train, dset_val=dset_val, train_mode=train_mode, scheduler=scheduler,
            model_encoder=model_encoder, model_projector=model_projector, model_inference=model_inference,
            path_save=path_save, min_val_loss=min_val_loss, batch_size=batch_size_val, seed=seed,
            device=device, save_aux_fig_name=str(tt), box=box, factor_kmin_cut=factor_kmin_cut,
            **kwargs['val']
        )
        
    return min_val_loss


def train_single_epoch(
    dset_train, train_mode, # "train_CL", "train_inference_from_latents", "train_inference_fully_supervised", or "train_CL_and_inference"
    optimizer,
    model_encoder, model_projector=None, model_inference=None,
    NN_batches_per_epoch=10, batch_size=16, clip_grad_norm=None,
    seed_mode="random", seed=0, seed0=0,
    device="cuda", NN_print_progress=None, save_aux_fig_name_epoch=None, box=2000, factor_kmin_cut=4,
    **kwargs
    ):
    """
    Trains a model for a single epoch.

    Parameters:
    - dset_train: Training dataset.
    - train_mode (str): Mode of training.
    - optimizer: Optimizer for training.
    - model_encoder (torch.nn.Module): Encoder model.
    - model_projector (torch.nn.Module, optional): Projector model.
    - model_inference (torch.nn.Module, optional): Inference model.
    - NN_batches_per_epoch (int, optional): Number of batches per epoch.
    - batch_size (int, optional): Batch size.
    - clip_grad_norm (float, optional): Gradient clipping norm.
    - seed_mode (str, optional): Seed mode.
    - seed (int, optional): Seed value.
    - seed0 (int, optional): Initial seed value.
    - device (str, optional): Device to use for training.
    - save_aux_fig_name_epoch (str, optional): Auxiliary figure name for saving.
    - box (int, optional): Box size parameter.
    - factor_kmin_cut (int, optional): Factor kmin cut parameter.
    - **kwargs: Additional keyword arguments.
    """
    if NN_print_progress == None: NN_print_progress=10
    if NN_print_progress > NN_batches_per_epoch: NN_print_progress = NN_batches_per_epoch
    
    if train_mode == "train_CL":
        model_encoder.train(); model_projector.train()
    if train_mode == "train_inference_from_latents":
        model_encoder.eval(); model_inference.train()
    if train_mode == "train_inference_fully_supervised":
        model_encoder.train(); model_inference.train()
    if train_mode == "train_CL_and_inference":
        model_encoder.train(); model_projector.train(); model_inference.train()
    
    LOSS = {}
    for ii_batch in range(NN_batches_per_epoch):
        
        # draw batch from dataset
        if seed_mode == "random": seed = datetime.datetime.now().microsecond %13037
        if seed_mode == "deterministic": seed = seed0 + ii_batch
        theta_true, xx, aug_params = dset_train(batch_size, seed=seed, to_torch=True, device=device, box=box, factor_kmin_cut=factor_kmin_cut)
        
        # compute loss
        LOSS = custom_loss_functions.compute_loss(
            theta_true, xx, aug_params,
            train_mode=train_mode, 
            model_encoder=model_encoder, model_projector=model_projector, model_inference=model_inference,
            CL_loss=kwargs['CL_loss'],
            inference_loss=kwargs['inference_loss'],
            save_aux_fig_name=None, # save_aux_fig_name_epoch+'_'+str(ii_batch) <-- to save figures during training... BE CAREFULL, they are going to be A LOT
            **kwargs['loss_hyperparameters']
        )
        
        # ipdb.set_trace()  # Add this line to set an ipdb breakpoint
        
        # perform backpropagation (update weights of the model)
        optimizer.zero_grad()
        LOSS['loss'].backward()
        if isinstance(clip_grad_norm, float):
            if train_mode == "train_CL":
                torch.nn.utils.clip_grad_norm_([*model_encoder.parameters(), *model_projector.parameters()], max_norm=clip_grad_norm)
            if train_mode == "train_inference_from_latents":
                torch.nn.utils.clip_grad_norm_([*model_inference.parameters()], max_norm=clip_grad_norm)
            if train_mode == "train_inference_fully_supervised":
                torch.nn.utils.clip_grad_norm_([*model_encoder.parameters(), *model_inference.parameters()], max_norm=clip_grad_norm)
            if train_mode == "train_CL_and_inference":
                torch.nn.utils.clip_grad_norm_([*model_encoder.parameters(), *model_projector.parameters(), *model_inference.parameters()], max_norm=clip_grad_norm)
        optimizer.step()       
        
        # print progress
        if (ii_batch+1) % int(NN_batches_per_epoch/NN_print_progress) == 0:
            logging.info(f"loss: {LOSS['loss'].item():>7f} | batch: [{ii_batch+1:>5d}/{NN_batches_per_epoch:>5d}]")
        
        if device == "cuda":
            torch.cuda.empty_cache()
        
    return logging.info(f"\n---------- done train epoch ---------")
    

def eval_single_epoch(
    dset_train, dset_val, train_mode, # "train_CL", "train_inference_from_latents", "train_inference_fully_supervised", or "train_CL_and_inference"
    scheduler, model_encoder, model_projector=None, model_inference=None,
    path_save=None, min_val_loss=None, batch_size=None,
    device="cuda", seed=0, save_aux_fig_name=None, box=2000, factor_kmin_cut=4, **kwargs
    ):
    """
    Evaluates a model after training for one epoch.

    Parameters:
    - dset_train: Training dataset.
    - dset_val: Validation dataset.
    - train_mode (str): Mode of training.
    - scheduler: Learning rate scheduler.
    - model_encoder (torch.nn.Module): Encoder model.
    - model_projector (torch.nn.Module, optional): Projector model.
    - model_inference (torch.nn.Module, optional): Inference model.
    - path_save (str, optional): Path to save the model.
    - min_val_loss (float, optional): Minimum validation loss.
    - batch_size (int, optional): Batch size for validation.
    - device (str, optional): Device to use for evaluation.
    - seed (int, optional): Seed value.
    - save_aux_fig_name (str, optional): Auxiliary figure name for saving.
    - box (float, optional): Box size parameter.
    - factor_kmin_cut (int, optional): Factor kmin cut parameter.
    - **kwargs: Additional keyword arguments.

    Returns:
    - min_val_loss (float): Updated minimum validation loss.
    - train_loss (float): Training loss.
    - val_loss (float): Validation loss.
    """
    if batch_size == None:
        batch_size=dset_val.total_cosmos_avail

    train_loss = eval_dataset(
        dset_train, train_mode, batch_size,
        model_encoder=model_encoder, model_projector=model_projector, model_inference=model_inference,
        seed=seed, device=device, box=box, factor_kmin_cut=factor_kmin_cut,
        save_aux_fig_name=None, # <-- replace by save_aux_fig_name if you want to print validation plots
        **kwargs
    )
    
    val_loss = eval_dataset(
        dset_val, train_mode, batch_size,
        model_encoder=model_encoder, model_projector=model_projector, model_inference=model_inference,
        seed=seed, device=device, box=box, factor_kmin_cut=factor_kmin_cut,
        save_aux_fig_name=None, # <-- replace by save_aux_fig_name if you want to print validation plots
        **kwargs
    )
    
    if min_val_loss == None:
        min_val_loss = val_loss['loss']
        if path_save!=None:
            if not os.path.exists(path_save):
                os.makedirs(path_save)
            ff = open(os.path.join(path_save, 'register.txt'), 'w')
            ff.write('%.4e %.4e\n'%(train_loss['loss'], val_loss['loss']))
            ff.close()
            logging.info(f"Saving Model from Epoch-0")
            if train_mode == "train_CL":
                torch.save(model_encoder.state_dict(), os.path.join(path_save, 'model_encoder.pt'))
            if train_mode == "train_inference_from_latents":
                torch.save(model_inference.state_dict(), os.path.join(path_save, 'model_inference.pt'))
            if train_mode == "train_inference_fully_supervised":
                torch.save(model_encoder.state_dict(), os.path.join(path_save, 'model_encoder.pt'))
                torch.save(model_inference.state_dict(), os.path.join(path_save, 'model_inference.pt'))
            if train_mode == "train_CL_and_inference":
                torch.save(model_encoder.state_dict(), os.path.join(path_save, 'model_encoder.pt'))
                torch.save(model_inference.state_dict(), os.path.join(path_save, 'model_inference.pt'))
    else:
        if path_save!=None:
            ff = open(os.path.join(path_save, 'register.txt'), 'a')
            ff.write('%.4e %.4e\n'%(train_loss['loss'], val_loss['loss']))
            ff.close()
    
    logging.info(f"min_val_loss = {min_val_loss:>7f}")
    logging.info(f"train_loss = {train_loss['loss'].item():>7f}")
    logging.info(f"val_loss = {val_loss['loss'].item():>7f}")
        
    if val_loss['loss'] < min_val_loss:
        min_val_loss = val_loss['loss']
        logging.info(f"Saving Model")
        if path_save!=None:
            if train_mode == "train_CL":
                torch.save(model_encoder.state_dict(), os.path.join(path_save, 'model_encoder.pt'))
            if train_mode == "train_inference_from_latents":
                torch.save(model_inference.state_dict(), os.path.join(path_save, 'model_inference.pt'))
            if train_mode == "train_inference_fully_supervised":
                torch.save(model_encoder.state_dict(), os.path.join(path_save, 'model_encoder.pt'))
                torch.save(model_inference.state_dict(), os.path.join(path_save, 'model_inference.pt'))
            if train_mode == "train_CL_and_inference":
                torch.save(model_encoder.state_dict(), os.path.join(path_save, 'model_encoder.pt'))
                torch.save(model_inference.state_dict(), os.path.join(path_save, 'model_inference.pt'))
                
    scheduler.step(val_loss['loss'])
    
    logging.info(f"\n--------- done eval epoch --------")
    
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return min_val_loss, train_loss, val_loss


def eval_dataset(
        dset, train_mode, batch_size,
        model_encoder, model_projector=None, model_inference=None,
        seed=0, device="cuda", save_aux_fig_name=None, box=2000, factor_kmin_cut=4,
        **kwargs
    ):
    """
    Evaluates a dataset and returns the computed loss.

    Parameters:
    - dset: Dataset to evaluate.
    - train_mode (str): Mode of training.
    - batch_size (int): Batch size for evaluation.
    - model_encoder (torch.nn.Module): Encoder model.
    - model_projector (torch.nn.Module, optional): Projector model.
    - model_inference (torch.nn.Module, optional): Inference model.
    - seed (int, optional): Seed value.
    - device (str, optional): Device to use for evaluation.
    - save_aux_fig_name (str, optional): Auxiliary figure name for saving.
    - box (float, optional): Box size parameter.
    - factor_kmin_cut (int, optional): Factor kmin cut parameter.
    - **kwargs: Additional keyword arguments for compute_loss.

    Returns:
    - LOSS (dict): Dictionary containing the output from compute_loss.
    """
    # draw batch from dataset
    theta_true, xx, aug_params = dset(batch_size, seed=seed, to_torch=True, device=device, box=box, factor_kmin_cut=factor_kmin_cut)
    
    # obtain model predictions & compute loss
    if train_mode == "train_CL":
        model_encoder.eval(); model_projector.eval()
    if train_mode == "train_inference_from_latents":
        model_inference.eval(); model_encoder.eval()
    if train_mode == "train_inference_fully_supervised":
        model_encoder.eval(); model_inference.eval()
    if train_mode == "train_CL_and_inference":
        model_encoder.eval(); model_projector.eval(); model_inference.eval()

    with torch.no_grad():
        LOSS = custom_loss_functions.compute_loss(
            theta_true, xx, aug_params,
            train_mode=train_mode, 
            model_encoder=model_encoder,
            model_projector=model_projector,
            model_inference=model_inference,
            CL_loss=kwargs['CL_loss'],
            inference_loss=kwargs['inference_loss'],
            save_aux_fig_name=save_aux_fig_name,
            **kwargs['loss_hyperparameters']
        )
    
    return LOSS