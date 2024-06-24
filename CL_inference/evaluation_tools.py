import os, sys
import yaml
from pathlib import Path
import torch
import numpy as np

from . import nn_tools
from . import data_tools
from . import custom_loss_functions
from . import train_tools
from . import plot_utils


def load_configs_file(models_path, selected_sweeps):

    configs = {}
    for ii, sweep_name in enumerate(selected_sweeps):
        if sweep_name == "manual-sweep-0":
            path_to_config=models_path + "/"+ sweep_name
            config_file_name = "config.yaml"
            configs[sweep_name] = train_tools.load_config_file(
                path_to_config=path_to_config,
                config_file_name=config_file_name
            )
        else:
            path_to_config=models_path+"/"+sweep_name
            configs[sweep_name] = load_config_file_wandb_format(
                path_to_config=path_to_config,
                config_file_name="config.yaml"
            )
            
    list_assert_compatible_keys = [
        "normalize","CL_loss","NN_params_out","NN_augs_batch","add_noise_Pk","inference_loss",
        "input_encoder","kmax","list_model_names","load_encoder_model_path","normalize",
        "output_encoder","output_projector","path_load","path_save","seed_mode", "train_mode"
    ]

    for ii, key in enumerate(list_assert_compatible_keys):
        tmp_list = []
        for jj, sweep_name in enumerate(selected_sweeps):
            tmp_list.append(configs[sweep_name][key])
        assert all(
            x==tmp_list[0] for x in tmp_list
        ), "ERROR for " + sweep_name + " - key: " + key + ". Not all config files share the same value: " + str(tmp_list)
            
        return configs


def load_config_file_wandb_format(path_to_config, config_file_name):
    config_wandb = yaml.safe_load(Path(os.path.join(path_to_config, config_file_name)).read_text())
    config = {}
    for ii, key in enumerate(config_wandb.keys()):
        try:
            config[key] = config_wandb[key]['value']
        except:
            print("Warning for " + path_to_config.split('/')[-1] + " - not value for key", key)
    return config
    
    
def reload_models(models_path, evalute_mode, configs, device):

    models_encoder = {}
    models_inference = {}

    for ii, sweep_name in enumerate(configs.keys()):
        
        config = configs[sweep_name]
        
        main_name = models_path.split('/')[-1]
        if ("only_inference" in main_name) and ("CL" in main_name):
            load_encoder_model_path = config["load_encoder_model_path"]
        else:
            load_encoder_model_path = os.path.join(models_path, sweep_name, "model_encoder.pt")

        if evalute_mode != "eval_CL":
            load_inference_model_path = os.path.join(models_path, sweep_name, "model_inference.pt")
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
        print("Loaded model encoder:", load_encoder_model_path.split('/')[-2])

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
            print("Loaded model inference:", load_inference_model_path.split('/')[-2])
    
    return models_encoder, models_inference
    
    
def compute_dataset_results(config, sweep_name_load_norm_dset, list_model_names, models_encoder, models_inference, dset_key="TEST", use_all_dataset_augs_ordered=True, indexes_cosmo=None, indexes_augs=None):
    
    if next(models_encoder[list(models_encoder.keys())[0]].parameters()).is_cuda: device = "cuda"
    else: device = "cpu"
    
    if config['include_baryon_params']:
        loaded_theta, loaded_xx, loaded_aug_params, len_models = data_tools.load_stored_data(
            path_load=os.path.join(config['path_load'], dset_key),
            list_model_names=list_model_names,
            return_len_models=True,
            include_baryon_params=config['include_baryon_params']
        )
    else:
        loaded_theta, loaded_xx, len_models = data_tools.load_stored_data(
            path_load=os.path.join(config['path_load'], dset_key),
            list_model_names=list_model_names,
            return_len_models=True,
            include_baryon_params=config['include_baryon_params']
        )
        loaded_aug_params = None

    dset = data_tools.data_loader(
        loaded_theta,
        loaded_xx,
        aug_params=loaded_aug_params,
        normalize=config['normalize'],
        path_load_norm = os.path.join(config['path_save'], sweep_name_load_norm_dset),
        NN_augs_batch = np.sum(len_models),
        add_noise_Pk=config['add_noise_Pk'],
        kmax=config['kmax']
    )
    if (type(indexes_cosmo) == type(np.array([]))) and (type(indexes_augs) != type(np.array([]))):
        indexes_augs=np.repeat(np.arange(dset.NN_augs)[np.newaxis], repeats=len(indexes_cosmo), axis=0)
    
    theta_true, xx, aug_params, indexes_cosmo, indexes_augs = dset(
        0, seed=0, to_torch=True, device=device, use_all_dataset_augs_ordered=use_all_dataset_augs_ordered, indexes_cosmo=indexes_cosmo, indexes_augs=indexes_augs, return_indexes_sampled=True
    )
    
    theta_true = torch.repeat_interleave(theta_true, xx.shape[1], axis=0)
    if aug_params is not None:
        aug_params = torch.reshape(aug_params, (aug_params.shape[0]*aug_params.shape[1], aug_params.shape[-1]))
        theta_true = torch.concatenate((theta_true, aug_params), axis=-1)
    theta_true = theta_true.cpu().detach().numpy()
        
    NN_params_out = config["NN_params_out"]
    tmp_xx = torch.reshape(xx, (np.prod(xx.shape[:2]),) + (xx.shape[-1],))
    thetas_pred = np.zeros((len(models_encoder.keys()), tmp_xx.shape[0], NN_params_out))
    Covs = np.zeros((len(models_encoder.keys()), tmp_xx.shape[0], NN_params_out, NN_params_out))
    hh = {}
    for ii, sweep_name in enumerate(models_encoder.keys()):
        hh[sweep_name] = models_encoder[sweep_name](tmp_xx.contiguous())
        
        if models_inference[sweep_name] != None:
            yy = models_inference[sweep_name](hh[sweep_name].contiguous())

            theta_pred, cov_pred = yy[:, :theta_true.shape[-1]], yy[:, theta_true.shape[-1]:]
            Cov = custom_loss_functions.vector_to_Cov(cov_pred).to(device)

            thetas_pred[ii] = theta_pred.cpu().detach().numpy()
            Covs[ii] = Cov.cpu().detach().numpy()
            
        hh[sweep_name] = hh[sweep_name].cpu().detach().numpy()
        hh[sweep_name] = np.reshape(hh[sweep_name], (indexes_augs.shape[0], indexes_augs.shape[1], hh[sweep_name].shape[-1]))
    
    xx = xx.cpu().detach().numpy()
    
    if models_inference[sweep_name] != None:
        theta_pred = np.mean(thetas_pred, axis=0)
        Cov = np.mean(Covs, axis=0)
        theta_pred = np.reshape(theta_pred, (indexes_augs.shape[0], indexes_augs.shape[1], theta_true.shape[-1]))
        Cov = np.reshape(Cov, (indexes_augs.shape[0], indexes_augs.shape[1], theta_true.shape[-1], theta_true.shape[-1]))
    else:
        theta_pred = np.zeros((indexes_augs.shape[0], indexes_augs.shape[1], theta_true.shape[-1]))
        Cov= np.zeros((indexes_augs.shape[0], indexes_augs.shape[1], theta_true.shape[-1], theta_true.shape[-1]))
        
    theta_true = np.reshape(theta_true, theta_pred.shape)
    return xx, hh, theta_true, theta_pred, Cov, len_models
    

def compute_bias_hist(true, pred, err, min_x=-6, max_x=6, bins=60):
    
    tmp_hist = (pred - true) / err
    counts, bin_edges = np.histogram(tmp_hist, bins=bins, range=(min_x, max_x))
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    counts = np.insert(counts, 0, np.sum(tmp_hist<min_x), axis=0)
    counts = np.insert(counts, len(counts), np.sum(tmp_hist>max_x), axis=0)

    return bin_edges, bin_centers, counts, true.shape[0]
    
    
def compute_bias_hist_augs(true, pred, Cov, min_x=-6, max_x=6, bins=60):
    
    NN_params = pred.shape[-1]
    NN_augs = pred.shape[1]
    bin_edges = np.zeros((NN_params, NN_augs, bins+1))
    bin_centers = np.zeros((NN_params, NN_augs, bins))
    y_hists = np.zeros((NN_params, NN_augs, bins+2))
    NN_points = np.zeros((NN_params, NN_augs))
    for ii_cosmo in range(NN_params):
        for ii_aug in range(NN_augs):
            tmp_true = true[:, ii_aug, ii_cosmo]
            tmp_pred = pred[:, ii_aug, ii_cosmo]
            tmp_err = np.sqrt(Cov[:, ii_aug, ii_cosmo, ii_cosmo])
            bin_edges[ii_cosmo, ii_aug], bin_centers[ii_cosmo, ii_aug], y_hists[ii_cosmo, ii_aug], NN_points[ii_cosmo, ii_aug] = compute_bias_hist(
                tmp_true, tmp_pred, tmp_err, min_x=min_x, max_x=max_x, bins=bins
            )

    return bin_edges, bin_centers, y_hists, NN_points


def compute_err_hist_augs(Cov, max_x, min_x=[0,0,0,0,0,0,0,0,0,0,0,0], bins=60):
    
    NN_params = Cov.shape[-1]
    NN_augs = Cov.shape[1]
    bin_edges = np.zeros((NN_params, NN_augs, bins+1))
    bin_centers = np.zeros((NN_params, NN_augs, bins))
    y_hists = np.zeros((NN_params, NN_augs, bins+2))
    median = np.zeros((NN_params, NN_augs))
    std = np.zeros((NN_params, NN_augs))
    for ii_cosmo in range(NN_params):
        for ii_aug in range(NN_augs):            
            tmp_hist = 2*np.sqrt(Cov[:, ii_aug, ii_cosmo, ii_cosmo])
            counts, bin_edges[ii_cosmo, ii_aug] = np.histogram(tmp_hist, bins=bins, range=(min_x[ii_cosmo], max_x[ii_cosmo]))
            bin_centers[ii_cosmo, ii_aug] = (bin_edges[ii_cosmo, ii_aug][1:] + bin_edges[ii_cosmo, ii_aug][:-1])/2
            counts = np.insert(counts, 0, np.sum(tmp_hist<min_x[ii_cosmo]), axis=0)
            y_hists[ii_cosmo, ii_aug] = np.insert(counts, len(counts), np.sum(tmp_hist>max_x[ii_cosmo]), axis=0)
            median[ii_cosmo, ii_aug] = np.median(tmp_hist)
            std[ii_cosmo, ii_aug] = np.std(tmp_hist)
            
    return bin_edges, bin_centers, y_hists, median, std
    
    
def compute_bias_and_errorbar_stats(
    config,
    sweep_name_load_norm_dset,
    list_model_names,
    models_encoder,
    models_inference,
    NN_params,
    save_root,
    thresholds_bias,
    NN_bins_hist = 60,
    NN_bins_hist_err = 60,
    NN_avail_cosmo_test = 2048,
    NN_split = 20,
    max_err_hist=[0.05, 0.012, 0.12, 0.042, 0.06, 3.2, 1.5, 1.5, 3., .4, 1., 3.]
):
    
    indexes_cosmos = np.arange(NN_avail_cosmo_test)
    np.random.shuffle(indexes_cosmos)
    indexes_cosmos_groups = np.split(indexes_cosmos, (np.arange(NN_split)+1) * int(NN_avail_cosmo_test/NN_split))[:-1]

    fraction_biased = np.zeros((len(indexes_cosmos_groups), len(list_model_names), NN_params, len(thresholds_bias)))
    NN_points = np.zeros((len(indexes_cosmos_groups), len(list_model_names), NN_params))
    bin_centers = np.zeros((len(indexes_cosmos_groups), len(list_model_names), NN_params, NN_bins_hist+2))
    y_hists = np.zeros((len(indexes_cosmos_groups), len(list_model_names), NN_params, NN_bins_hist+2))
    bin_centers_err = np.zeros((len(indexes_cosmos_groups), len(list_model_names), NN_params, NN_bins_hist_err+2))
    y_hists_err = np.zeros((len(indexes_cosmos_groups), len(list_model_names), NN_params, NN_bins_hist_err+2))
    median_err = np.zeros((len(indexes_cosmos_groups), len(list_model_names), NN_params))
    std_err = np.zeros((len(indexes_cosmos_groups), len(list_model_names), NN_params))

    ii_aug = 0
    for ii_model_name, list_model_name in enumerate(list_model_names):
        for ii, indexes_cosmos in enumerate(indexes_cosmos_groups):
            xx, hh, theta_true, theta_pred, Cov, len_models = compute_dataset_results(
                config,
                sweep_name_load_norm_dset,
                list_model_names=[list_model_name],
                models_encoder=models_encoder,
                models_inference=models_inference,
                dset_key="TEST",
                use_all_dataset_augs_ordered=False,
                indexes_cosmo=indexes_cosmos
            )

            theta_true = np.reshape(theta_true, (theta_true.shape[0]*theta_true.shape[1], theta_true.shape[-1]))[:,np.newaxis]
            theta_pred = np.reshape(theta_pred, (theta_pred.shape[0]*theta_pred.shape[1], theta_pred.shape[-1]))[:,np.newaxis]
            Cov = np.reshape(Cov, (Cov.shape[0]*Cov.shape[1], Cov.shape[-2], Cov.shape[-1]))[:,np.newaxis]

            NN_samples = xx.shape[0]*xx.shape[1]

            # --------------------------- compute_bias_hist_augs --------------------------- #

            bin_edges, tmp_bin_centers, tmp_y_hists, tmp_NN_points = compute_bias_hist_augs(
                theta_true, theta_pred, Cov, min_x=-6, max_x=6, bins=NN_bins_hist
            )
            tmp_bin_centers = np.insert(tmp_bin_centers, 0, bin_edges[..., 0], axis=-1)
            tmp_bin_centers = np.insert(tmp_bin_centers, bin_edges.shape[-1], bin_edges[..., -1], axis=-1)

            fraction_biased_list = np.zeros((tmp_y_hists.shape[0], len(thresholds_bias)))
            for jj in range(len(thresholds_bias)):
                mask = np.abs(tmp_bin_centers) > thresholds_bias[jj]
                for kk in range(tmp_y_hists.shape[0]):
                    fraction_biased_list[kk, jj] = np.sum(tmp_y_hists[kk,ii_aug][mask[kk,ii_aug]]) / tmp_NN_points[kk,ii_aug]

            fraction_biased[ii, ii_model_name] = fraction_biased_list
            bin_centers[ii, ii_model_name] = tmp_bin_centers[:,ii_aug]
            y_hists[ii, ii_model_name] = tmp_y_hists[:,ii_aug]
            NN_points[ii, ii_model_name] = tmp_NN_points[:,ii_aug]

            # --------------------------- compute_err_hist_augs --------------------------- #
                                                                                               
            bin_edges_err, tmp_bin_centers_err, tmp_y_hists_err, tmp_median_err, tmp_std_err = compute_err_hist_augs(
                Cov, max_x=max_err_hist, bins=NN_bins_hist_err
            )
            tmp_bin_centers_err = np.insert(tmp_bin_centers_err, 0, bin_edges_err[..., 0], axis=-1)
            tmp_bin_centers_err = np.insert(tmp_bin_centers_err, bin_edges_err.shape[-1], bin_edges_err[..., -1], axis=-1)
            
            bin_centers_err[ii, ii_model_name] = tmp_bin_centers_err[:,ii_aug]
            y_hists_err[ii, ii_model_name] = tmp_y_hists_err[:,ii_aug]
            median_err[ii, ii_model_name] = tmp_median_err[:,ii_aug]
            std_err[ii, ii_model_name] = tmp_std_err[:,ii_aug]

    np.save(save_root + "/fraction_biased_batches.npy", fraction_biased)
    np.save(save_root + "/NN_points_batches.npy", NN_points)
    np.save(save_root + "/bin_centers_batches.npy", bin_centers)
    np.save(save_root + "/y_hists_batches.npy", y_hists)
    np.save(save_root + "/bin_centers_err_batches.npy", bin_centers_err)
    np.save(save_root + "/y_hists_err_batches.npy", y_hists_err)
    np.save(save_root + "/median_err_batches.npy", median_err)
    np.save(save_root + "/std_err_batches.npy", std_err)
    
    return fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err
    
    
def load_numpy_stored_summaries(models_path, mode_name, names_load, fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors, default_colors=None):
    
    fraction_biased[mode_name] = {}
    NN_points[mode_name] = {}
    bin_centers[mode_name] = {}
    y_hists[mode_name] = {}
    bin_centers_err[mode_name] = {}
    y_hists_err[mode_name] = {}
    median_err[mode_name] = {}
    std_err[mode_name] = {}
    colors[mode_name] = {}
    
    for ii, key in enumerate(names_load.keys()):
        try:
            fraction_biased[mode_name][key]         = np.load(os.path.join(models_path, names_load[key], "fraction_biased_batches.npy"))
            NN_points[mode_name][key]               = np.load(os.path.join(models_path, names_load[key], "NN_points_batches.npy"))
            bin_centers[mode_name][key]             = np.load(os.path.join(models_path, names_load[key], "bin_centers_batches.npy"))
            y_hists[mode_name][key]                 = np.load(os.path.join(models_path, names_load[key], "y_hists_batches.npy"))
            bin_centers_err[mode_name][key]         = np.load(os.path.join(models_path, names_load[key], "bin_centers_err_batches.npy"))
            y_hists_err[mode_name][key]             = np.load(os.path.join(models_path, names_load[key], "y_hists_err_batches.npy"))
            median_err[mode_name][key]              = np.load(os.path.join(models_path, names_load[key], "median_err_batches.npy"))
            std_err[mode_name][key]                 = np.load(os.path.join(models_path, names_load[key], "std_err_batches.npy"))
            
            if default_colors != None:
                colors[mode_name][key]          = default_colors[key]
        except:
            print("ERROR loading", names_load[key])
    
    return fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors


def wrapper_load_numpy_stored_summaries(
        models_path,
        list_model_names = {
            "illustris-eagle"   : "illustris_eagle",
            "eagle-bahamas"     : "eagle_bahamas",
            "bahamas-illustris" : "bahamas_illustris",
            "v1-v2"             : "v1_v2",
            "v1-v3"             : "v1_v3",
            "v2-v3"             : "v2_v3",
            "f0-f1"             : "f0_f1",
            "f2-f3"             : "f2_f3",
            "f4-f5"             : "f4_f5",
            "f6-f7"             : "f6_f7",
            "f8-f9"             : "f8_f9"
        }
    ):
    
    fraction_biased = {}
    NN_points = {}
    bin_centers = {}
    y_hists = {}
    bin_centers_err = {}
    y_hists_err = {}
    median_err = {}
    std_err = {}
    colors = {}
    
    default_colors = plot_utils.colors_combined_dsets()
    
    fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors = load_numpy_stored_summaries(
        models_path, "all", {"all" : "only_inference_models_all_kmax_0.6"}, fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors, default_colors
    )
    
    dict_names_load = {}
    for ii, key in enumerate(list_model_names.keys()):
        dict_names_load[key] = "only_inference_models_" + list_model_names[key] + "_kmax_0.6"
    fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors = load_numpy_stored_summaries(
        models_path, "no CL", dict_names_load, fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors, default_colors
    )
    
    dict_names_load = {}
    for ii, key in enumerate(list_model_names.keys()):
        dict_names_load[key] = "only_inference_CL_Wein_models_" + list_model_names[key] + "_kmax_0.6"
    fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors = load_numpy_stored_summaries(
        models_path, "CL Wein", dict_names_load, fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors, default_colors
    )

    dict_names_load = {}
    for ii, key in enumerate(list_model_names.keys()):
        dict_names_load[key] = "only_inference_CL_VICReg_models_" + list_model_names[key] + "_kmax_0.6"
    fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors = load_numpy_stored_summaries(
        models_path, "CL VICReg", dict_names_load, fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors, default_colors
    )
    
    return fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors
    

def wrapper_load_numpy_stored_summaries_kcuts(models_path, kcuts=np.array([0.6, 0.2, -0.2, -0.6, -1.0, -1.4])):

    fraction_biased = {}
    NN_points = {}
    bin_centers = {}
    y_hists = {}
    bin_centers_err = {}
    y_hists_err = {}
    median_err = {}
    std_err = {}
    colors = {}

    default_colors = None
        
    dict_names_load = {}
    for ii, kcut in enumerate(kcuts):
        dict_names_load[str(kcut)] = "only_inference_also_baryons_models_all_kmax_" + str(kcut)
    fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors = load_numpy_stored_summaries(
        models_path, "all also baryons", dict_names_load, fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors, default_colors
    )

    dict_names_load = {}
    for ii, kcut in enumerate(kcuts):
        dict_names_load[str(kcut)] = "only_inference_models_all_kmax_" + str(kcut)
    fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors = load_numpy_stored_summaries(
        models_path, "all not baryons", dict_names_load, fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors, default_colors
    )

    dict_names_load = {}
    for ii, kcut in enumerate(kcuts):
        dict_names_load[str(kcut)] = "only_inference_models_illustris_eagle_kmax_" + str(kcut)
    fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors = load_numpy_stored_summaries(
        models_path, "no CL", dict_names_load, fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors, default_colors
    )

    dict_names_load = {}
    for ii, kcut in enumerate(kcuts):
        dict_names_load[str(kcut)] = "only_inference_CL_Wein_models_illustris_eagle_kmax_" + str(kcut)
    fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors = load_numpy_stored_summaries(
        models_path, "CL Wein", dict_names_load, fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors, default_colors
    )

    dict_names_load = {}
    for ii, kcut in enumerate(kcuts):
        dict_names_load[str(kcut)] = "only_inference_CL_VICReg_models_illustris_eagle_kmax_" + str(kcut)
    fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors = load_numpy_stored_summaries(
        models_path, "CL VICReg", dict_names_load, fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors
    )
    
    return fraction_biased, NN_points, bin_centers, y_hists, bin_centers_err, y_hists_err, median_err, std_err, colors
    