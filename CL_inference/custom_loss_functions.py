import torch
import math
import ipdb

def weinberger_loss(yy, delta_pull=0.5, delta_push=1.5, c_pull=1., c_push=1., c_reg=0.001, _epsilon=1e-8):
        
    # ----------------------- Compute positions cluster centers ----------------------- #

    cluster_centers = torch.sum(yy, axis=1)[:,None] / yy.shape[1]

    # ----------------------- Compute L_reg ----------------------- #

    cluster_centers_distance_origin = torch.sqrt(torch.sum(cluster_centers**2, axis=-1)+_epsilon)
    L_reg = torch.sum(cluster_centers_distance_origin) / yy.shape[0]

    # ----------------------- Compute L_pull ----------------------- #

    distances_to_cluster_centers = torch.sqrt(torch.sum((yy - cluster_centers.repeat(1, yy.shape[1], 1))**2, axis=-1)+_epsilon)
    hinged_distances_to_cluster_centers = distances_to_cluster_centers - delta_pull
    ind_L_pull_terms = torch.clip(
        hinged_distances_to_cluster_centers,
        0., torch.max(torch.FloatTensor([0., torch.max(hinged_distances_to_cluster_centers)]))
    )**2
    L_pull = torch.sum(ind_L_pull_terms)

    # ----------------------- Compute L_push ----------------------- #

    casted_cluster_centers = cluster_centers.repeat(1, yy.shape[0], 1)
    relative_posistions_cluster_centers = casted_cluster_centers - torch.transpose(casted_cluster_centers, 0, 1)
    distances_between_cluster_centers = torch.sqrt(torch.sum(relative_posistions_cluster_centers**2, axis=-1)+_epsilon)
    hinged_distances_between_cluster_centers = 2*delta_push - distances_between_cluster_centers
    ind_L_push_terms = torch.triu(torch.clip(
        hinged_distances_between_cluster_centers,
        0., 2.*delta_push
    )**2, diagonal=1)
    L_push = torch.sum(ind_L_push_terms)

    # ----------------------- total loss ----------------------- #
    
    loss = c_pull*L_pull + c_push*L_push + c_reg*L_reg
    
    return loss, L_pull, L_push, L_reg, cluster_centers
    
# the loss is taken from
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
# def nce_loss(z_a, z_b, temperature=0.1):
    # print('z_a', z_a.shape)
    # print('z_b', z_b.shape)
    # # Calculate cosine similarity
    # cos_sim = torch.functional.F.cosine_similarity(z_a, z_b, dim=-1)
    # print('cos_sim', cos_sim.shape)
    # # Mask out cosine similarity to itself
    # print('cos_sim.shape[0]', cos_sim.shape[0])
    # self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    # print('self_mask', self_mask.shape)
    # cos_sim.masked_fill_(self_mask, -9e15)
    # # Find positive example -> batch_size//2 away from the original example
    # pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # # InfoNCE loss
    # cos_sim = cos_sim / temperature
    # nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    # nll = nll.mean()
    # return nll
# def nce_loss(features, temperature=0.1):
    # # Calculate cosine similarity
    # cos_sim = torch.functional.F.cosine_similarity(features[:,None,:], features[None,:,:], dim=-1)
    # # Mask out cosine similarity to itself
    # self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    # cos_sim.masked_fill_(self_mask[..., None], -9e15)
    # # Find positive example -> batch_size//2 away from the original example
    # pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # # InfoNCE loss
    # cos_sim = cos_sim / temperature
    # nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    # nll = nll.mean()
    # return nll
    
    
## adapted from "generally intellient" team's code 
## https://generallyintelligent.com/open-source/2022-04-21-vicreg/
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_loss(z_a, z_b, inv_weight=25, var_weight=25, cov_weight=1):
    
    assert z_a.shape == z_b.shape and len(z_a.shape) == 2
    # invariance loss
    loss_inv = torch.functional.F.mse_loss(z_a, z_b)
    
    # variance loss
    std_z_a = torch.sqrt(z_a.var(dim=0) + 0.0001) 
    std_z_b = torch.sqrt(z_b.var(dim=0) + 0.0001) 
    loss_v_a = torch.mean(torch.functional.F.relu(1 - std_z_a))
    loss_v_b = torch.mean(torch.functional.F.relu(1 - std_z_b)) 
    loss_var = (loss_v_a + loss_v_b) 
    
    z_a = z_a - z_a.mean(dim=0)
    z_b = z_b - z_b.mean(dim=0)

    # covariance loss
    N, D = z_a.shape
    cov_z_a = ((z_a.T @ z_a) / (N - 1)) # DxD
    cov_z_b = ((z_b.T @ z_b) / (N - 1)) # DxD
    loss_cov = off_diagonal(cov_z_a).pow_(2).sum().div(D) 
    loss_cov += off_diagonal(cov_z_b).pow_(2).sum().div(D)
    
    weighted_inv = loss_inv * inv_weight
    weighted_var = loss_var * var_weight
    weighted_cov = loss_cov * cov_weight

    loss = weighted_inv + weighted_var + weighted_cov   
    return loss, weighted_inv, weighted_var, weighted_cov
    
    
def compute_loss(
    theta_true,
    xx,
    aug_params,
    train_mode, # "train_CL", "train_inference_from_latents", "train_inference_fully_supervised", or "train_CL_and_inference"
    model_encoder, model_projector=None, model_inference=None,
    CL_loss="VicReg",
    inference_loss="MSE",
    save_aux_fig_name=None,
    **kwargs
    ):
    
    if next(model_encoder.parameters()).is_cuda: device = "cuda"
    else: device = "cpu"
    
    if 'c_CL' in list(kwargs.keys()):
        c_CL = kwargs['c_CL']
        kwargs.pop('c_CL', None)
    else:
        c_CL = 1.
    if 'c_inference' in list(kwargs.keys()):
        c_inference = kwargs['c_inference']
        kwargs.pop('c_inference', None)
    else:
        c_inference = 1.
    
    len_batch = xx.shape[0]
    len_augs = xx.shape[1]
    len_batch_times_aug = len_batch * len_augs
    size_for_batch_resize = tuple([len_batch_times_aug,] + list(xx.shape[2:]))
    xx = torch.reshape(xx, size_for_batch_resize)
    
    theta_true = torch.repeat_interleave(theta_true, len_augs, axis=0)
    if aug_params is not None:
        aug_params = torch.reshape(aug_params, (len_batch_times_aug, aug_params.shape[-1]))
        theta_true = torch.concatenate((theta_true, aug_params), axis=-1)
    
    hh = model_encoder(xx.contiguous())

    if (train_mode == "train_CL") or (train_mode == "train_CL_and_inference"):
        zz = {}
        zz = model_projector(hh.contiguous())
        zz = torch.reshape(zz, tuple([len_batch, len_augs,] + list(zz.shape[1:])))

        if CL_loss == "VicReg":
            loss_CL, inv, var, cov = vicreg_loss(zz[:,0], zz[:,1], **kwargs)
        if CL_loss == "Weinberger":
            loss_CL, L_pull, L_push, L_reg, cluster_centers = weinberger_loss(zz, **kwargs)
        if CL_loss == "SimCLR":
            loss_CL = nce_loss(zz, **kwargs)

    if (train_mode == "train_inference_fully_supervised") or (train_mode == "train_inference_from_latents") or (train_mode == "train_CL_and_inference"):
        
        if inference_loss == "MSE":
            theta_pred = model_inference(hh.contiguous())
            loss_inference = torch.nn.MSELoss()(theta_pred, theta_true)        

        if inference_loss == "MultivariateNormal":
            yy = model_inference(hh.contiguous())
            theta_pred = yy[:, :theta_true.shape[-1]]
            Cov = vector_to_Cov(yy[:, theta_true.shape[-1]:]).to(device=device)
            # ipdb.set_trace()  # Add this line to set an ipdb breakpoint
            loss_inference = -torch.distributions.MultivariateNormal(loc=theta_pred, covariance_matrix=Cov).log_prob(theta_true).mean()
    
    LOSS = {}
    if train_mode == "train_CL":
        LOSS['loss'] = c_CL*loss_CL
        LOSS['loss'] = LOSS['loss'] / len_batch_times_aug
    if (train_mode == "train_inference_fully_supervised") or (train_mode == "train_inference_from_latents"):
        LOSS['loss'] = c_inference*loss_inference
        LOSS['loss'] = LOSS['loss'] / len_batch_times_aug
    if train_mode == "train_CL_and_inference":
        LOSS['loss'] = c_CL*loss_CL + c_inference*loss_inference
        LOSS['loss'] = LOSS['loss'] / len_batch_times_aug

    if save_aux_fig_name != None:
        
        path_save_figs = "/cosmos_storage/home/dlopez/Projects/CL_inference/models/aux/"
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        
        xx = torch.reshape(xx, tuple([len_batch, len_augs,] + list(xx.shape[1:])))
        xx = xx.cpu().detach().numpy()

        theta_true = torch.reshape(theta_true, tuple([len_batch, len_augs,] + [theta_true.shape[-1],]))
        theta_true = theta_true.cpu().detach().numpy()
        
        box=2000
        kmax=0.6
        kf = 2.0 * np.pi / box
        kmin=np.log10(4*kf)
        N_kk = int((kmax - kmin) / (8*kf))
        kk = np.logspace(kmin, kmax, num=N_kk)
        
        ii_cosmo = 0
        ii_aug = 0
        tmp_xx = xx[ii_cosmo, ii_aug]
        tmp_theta = theta_true[ii_cosmo, ii_aug]
        
        # ipdb.set_trace()  # Add this line to set an ipdb breakpoint
        
        import baccoemu
        emulator = baccoemu.Matter_powerspectrum()
        baccoemu_params = {'neutrino_mass':0.0, 'w0':-1.0, 'wa':0.0, 'expfactor':1}
        
        # baccoemu_params.update({
            # 'omega_cold'    : tmp_theta[0],
            # 'omega_baryon'  : tmp_theta[1],
            # 'hubble'        : tmp_theta[2],
            # 'ns'            : tmp_theta[3],
            # 'sigma8_cold'   : tmp_theta[4],
            # 'M_c'           : tmp_theta[5],
            # 'eta'           : tmp_theta[6],
            # 'beta'          : tmp_theta[7],
            # 'M1_z0_cen'     : tmp_theta[8],
            # 'theta_out'     : tmp_theta[9],
            # 'theta_inn'     : tmp_theta[10],
            # 'M_inn'         : tmp_theta[11]
        # })
        # _, pk = emulator.get_nonlinear_pk(k=kk, cold=False, baryonic_boost=True, **baccoemu_params)        
        
        from bacco.baryons import emu_pars
        hydro_key = "illustris"
        baccoemu_params.update({
            'omega_cold'    : tmp_theta[0],
            'omega_baryon'  : tmp_theta[1],
            'hubble'        : tmp_theta[2],
            'ns'            : tmp_theta[3],
            'sigma8_cold'   : tmp_theta[4],
            'M_c'           : emu_pars(model=hydro_key)['z_0.0']['M_c'],
            'eta'           : emu_pars(model=hydro_key)['z_0.0']['eta'],
            'beta'          : emu_pars(model=hydro_key)['z_0.0']['beta'],
            'M1_z0_cen'     : emu_pars(model=hydro_key)['z_0.0']['M1_z0_cen'],
            'theta_out'     : emu_pars(model=hydro_key)['z_0.0']['theta_out'],
            'theta_inn'     : emu_pars(model=hydro_key)['z_0.0']['theta_inn'],
            'M_inn'         : emu_pars(model=hydro_key)['z_0.0']['M_inn']
        })
        _, pk = emulator.get_nonlinear_pk(k=kk, cold=False, baryonic_boost=True, **baccoemu_params)
        
        # ipdb.set_trace()  # Add this line to set an ipdb breakpoint
        
        tmp_norm_mean = np.array([4.35428156, 4.36015748, 4.36475835, 4.36809262, 4.36996703,
       4.37021599, 4.36896846, 4.36613879, 4.36144863, 4.35472825,
       4.34614102, 4.3352687 , 4.32215564, 4.30703706, 4.28982078,
       4.27062916, 4.2494197 , 4.22663361, 4.20257918, 4.17778117,
       4.15311367, 4.12921769, 4.10593366, 4.08374837, 4.0638817 ,
       4.04567219, 4.02784911, 4.0101774 , 3.99139571, 3.9695196 ,
       3.94324754, 3.91175207, 3.8747995 , 3.83350649, 3.79102707,
       3.75025135, 3.71439744, 3.68423313, 3.65844449, 3.63342863,
       3.60474155, 3.56870909, 3.52807497, 3.48659489, 3.44900147,
       3.41697402, 3.38686682, 3.35388105, 3.318306  , 3.28178307,
       3.24805839, 3.21606138, 3.18396789, 3.15089716, 3.11888847,
       3.08766333, 3.05696919, 3.02632348, 2.99654576, 2.96703171,
       2.93820136, 2.9098578 , 2.88186525, 2.85458741, 2.8278452 ,
       2.80117623, 2.7751666 , 2.74904262, 2.72357455, 2.69772721,
       2.67225013, 2.64659014, 2.62085942, 2.59465684, 2.56822441,
       2.54122955, 2.51372478, 2.48547934, 2.45656852, 2.4266999 ,
       2.39601406, 2.36431746, 2.33164247, 2.29788593, 2.26313773,
       2.22726035, 2.1903348 , 2.15233081, 2.1133777 , 2.07354798,
       2.03278398, 1.99109971, 1.9485184 , 1.90515954, 1.86096751,
       1.81617145, 1.77075056, 1.72479352, 1.67839099])
      
        tmp_norm_std = np.array([0.16138511, 0.15872159, 0.1559162 , 0.15297732, 0.1498267 ,
       0.14648881, 0.14295333, 0.13926021, 0.13534151, 0.13122879,
       0.12693179, 0.12236604, 0.11756748, 0.11264602, 0.10752461,
       0.10239087, 0.09724627, 0.09221445, 0.08747436, 0.08313844,
       0.0792702 , 0.07612923, 0.07373762, 0.07203453, 0.07119085,
       0.07083978, 0.07050079, 0.06999126, 0.06902798, 0.06720831,
       0.06448631, 0.06116831, 0.05768431, 0.05470236, 0.05268991,
       0.05179234, 0.05175027, 0.05240877, 0.05304853, 0.05284797,
       0.05188037, 0.0507271 , 0.05048029, 0.05089727, 0.05175044,
       0.05235522, 0.05235769, 0.052329  , 0.05302063, 0.05404826,
       0.05490246, 0.05549419, 0.05625115, 0.05720411, 0.05821663,
       0.05911676, 0.06005526, 0.06103708, 0.06207812, 0.06318785,
       0.06426805, 0.06537303, 0.06651713, 0.06769751, 0.06887531,
       0.07003323, 0.07109655, 0.07215754, 0.07318867, 0.07416905,
       0.07509536, 0.0759668 , 0.07680398, 0.07758599, 0.07831847,
       0.07900815, 0.07967418, 0.08030164, 0.0809208 , 0.08154053,
       0.08217686, 0.08286296, 0.08360377, 0.08442959, 0.0853392 ,
       0.08635807, 0.08746646, 0.08868993, 0.0900507 , 0.09154191,
       0.09313648, 0.09472103, 0.09638327, 0.09793211, 0.09950182,
       0.10086096, 0.10213803, 0.10317222, 0.10399069])

        fig, ax = mpl.pyplot.subplots(1,1,figsize=(7,6))
        ax.set_ylabel(r'$\mathrm{norm}\left(P(k) \left[ \left(h^{-1} \mathrm{Mpc}\right)^{3} \right]\right)$')
        ax.set_xlabel(r'$k - \mathrm{index} \left[ adim \right]$')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.plot(np.arange(tmp_xx.shape[-1]), tmp_xx, c='k', linestyle=':', lw=2, marker=None, ms=2)
        ax.plot(np.arange(len(kk)), (np.log10(pk) - tmp_norm_mean ) / tmp_norm_std, c='k', linestyle='-', lw=1.5, marker=None, ms=2, alpha=0.7)
        plt.tight_layout()
        plt.show()
        fig.savefig(path_save_figs + "Pk_" + save_aux_fig_name + ".png")
        plt.close(fig)
        
        ipdb.set_trace()  # Add this line to set an ipdb breakpoint
        
        # fig, ax = plot_utils.simple_plot()
        # ax.set_xscale('log')
        # colors = plot_utils.get_N_colors(xx_plot.shape[0], mpl.colormaps['prism'])
        # for ii in range(xx_plot.shape[0]):
            # ax.plot(xx_plot[ii].T, c=colors[ii], lw=0.5)
        # fig.savefig(path_save_figs + "train_" + save_aux_fig_name + ".png")
        # plt.tight_layout()
        # plt.show()
        # plt.close(fig)

        # if (train_mode == "train_CL") or (train_mode == "train_CL_and_inference"):
            # hh_plot = hh.cpu().detach().numpy()
            # hh_plot = np.reshape(hh_plot, tuple([len_batch, len_augs,] + list(hh_plot.shape[1:])))
            # colors = plot_utils.get_N_colors(hh_plot.shape[0], mpl.colormaps['prism'])
            # fig, ax = plot_utils.simple_plot()
            # for ii in range(hh_plot.shape[0]):
                # ax.scatter(hh_plot[ii,:,0], hh_plot[ii,:,1], c=colors[ii], lw=1)
            # fig.savefig(path_save_figs + "pseudo_" + save_aux_fig_name + ".png")
            # plt.tight_layout()
            # plt.show()
            # plt.close(fig)

        # if (train_mode == "train_inference_fully_supervised") or (train_mode == "train_inference_from_latents") or (train_mode == "train_CL_and_inference"):
            # theta_true_plot = theta_true.cpu().detach().numpy()
            # theta_pred_plot = theta_pred.cpu().detach().numpy()
            # theta_true_plot = np.reshape(theta_true_plot, tuple([len_batch, len_augs,] + [theta_true_plot.shape[-1],]))
            # theta_pred_plot = np.reshape(theta_pred_plot, tuple([len_batch, len_augs,] + [theta_pred_plot.shape[-1],]))
            
            # if inference_loss == "MSE":
                # colors = plot_utils.get_N_colors(len_batch, mpl.colormaps['prism'])
                # fig, axs = plt.subplots(1, theta_pred_plot.shape[-1], figsize=(5.2*theta_pred_plot.shape[-1], 5))
                # for ii in range(len_batch):
                    # for ii_cosmo_param in range(theta_pred_plot.shape[-1]):
                        # ax = axs[ii_cosmo_param]            
                        # ax.scatter(
                            # theta_true_plot[ii, :, ii_cosmo_param], theta_pred_plot[ii, :, ii_cosmo_param],
                            # color=colors[ii], alpha=1., marker ='o', s=6
                        # )
                        # ymax = np.nanmax(theta_true_plot[...,ii_cosmo_param])
                        # ymin = np.nanmin(theta_true_plot[...,ii_cosmo_param])
                        # tmp_xx = np.linspace(ymin, ymax, 2)
                        # ax.plot(tmp_xx, tmp_xx, c='k', lw=2, ls='-', alpha=1)
                        # ax.set_xlabel(r'True')
                    # axs[0].set_ylabel(r'Pred')
                # plt.tight_layout()
                # plt.show()
                # fig.savefig(path_save_figs + "inference_" + save_aux_fig_name + ".png")
                # plt.close(fig)
                
            # if inference_loss == "MultivariateNormal":
                # Cov_plot = Cov.cpu().detach().numpy()
                # Cov_plot = np.reshape(Cov_plot, tuple([len_batch, len_augs,] + list(Cov.shape[1:])))
                
                # colors = plot_utils.get_N_colors(len_batch, mpl.colormaps['prism'])
                # fig, axs = plt.subplots(1, theta_pred_plot.shape[-1], figsize=(5.2*theta_pred_plot.shape[-1], 5))
                # for ii in range(len_batch):
                    # for ii_cosmo_param in range(theta_pred_plot.shape[-1]):
                        # ax = axs[ii_cosmo_param]            
                        # ax.scatter(
                            # theta_true_plot[ii, :, ii_cosmo_param], theta_pred_plot[ii, :, ii_cosmo_param],
                           # color=colors[ii], alpha=1., marker ='o', s=6
                        # )
                        # ax.errorbar(
                            # theta_true_plot[ii, :, ii_cosmo_param], theta_pred_plot[ii, :, ii_cosmo_param],
                            # yerr=np.sqrt(Cov_plot[ii, :, ii_cosmo_param, ii_cosmo_param]),
                            # c=colors[ii], ls='', capsize=2, alpha=0.6, elinewidth=1
                        # )
                        # ymax = np.nanmax(theta_true_plot[...,ii_cosmo_param])
                        # ymin = np.nanmin(theta_true_plot[...,ii_cosmo_param])
                        # tmp_xx = np.linspace(ymin, ymax, 2)
                        # ax.plot(tmp_xx, tmp_xx, c='k', lw=2, ls='-', alpha=1)
                        # ax.set_xlabel(r'True')
                    # axs[0].set_ylabel(r'Pred')
                # plt.tight_layout()
                # plt.show()
                # fig.savefig(path_save_figs + "inference_" + save_aux_fig_name + ".png")
                # plt.close(fig)
                
    return LOSS
    

def vector_to_Cov(vec, device="cuda"):
    """ Convert unconstrained vector into a positive-diagonal, symmetric covariance matrix
        by converting to cholesky matrix, then doing Cov = L @ L^T 
        (https://en.wikipedia.org/wiki/Cholesky_decomposition)
    """
    D = int((-1.0 + math.sqrt(1.0 + 8.0 * vec.shape[-1])) / 2.0)  # Infer dimensionality; D * (D + 1) / 2 = n_tril
    B = vec.shape[0]  # Batch dim
    
    # Get indices of lower-triangular matrix to fill
    tril_indices = torch.tril_indices(row=D, col=D, offset=0)
    
    # Fill lower-triangular Cholesky matrix
    L = torch.zeros((B, D, D)).to(device=device)
    mask1 = torch.zeros(L.shape, device=L.device, dtype=torch.bool)
    mask1[:, tril_indices[0], tril_indices[1]] = True
    L = L.masked_scatter(mask1, vec)
    
    # Enforce positive diagonals
    positive_diags = torch.nn.Softplus()(torch.diagonal(L, dim1=-1, dim2=-2))
    
    mask2 = torch.zeros(L.shape, device=L.device, dtype=torch.bool)
    mask2[:, range(L.shape[-1]), range(L.shape[-2])] = True
    L = L.masked_scatter(mask2, positive_diags)
    
    # Cov = L @ L^T 
    Cov = torch.einsum("bij, bkj ->bik",L, L)

    return Cov