import torch
import math

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
    train_mode, # "train_CL", "train_inference_from_latents", "train_inference_fully_supervised", or "train_CL_and_inference"
    model_encoder, model_projector=None, model_inference=None,
    CL_loss="VicReg",
    inference_loss="MSE",
    save_aux_fig_name=None,
    **kwargs
    ):
    
    LOSS = {}
    
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
    
    hh = {}
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
            loss_inference = -torch.distributions.MultivariateNormal(loc=theta_pred, covariance_matrix=Cov).log_prob(theta_true).mean()
        
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

        # xx = torch.reshape(xx, tuple([len_batch, len_augs,] + list(xx.shape[1:])))
        # xx_plot = xx.cpu().detach().numpy()
        # fig, ax = plot_utils.simple_plot()
        # ax.set_xscale('log')
        # colors = plot_utils.get_N_colors(xx_plot.shape[0], mpl.colormaps['prism'])
        # for ii in range(xx_plot.shape[0]):
            # ax.plot(xx_plot[ii].T, c=colors[ii], lw=0.5)
        # fig.savefig(path_save_figs + "train_" + save_aux_fig_name + ".png")
        # plt.tight_layout()
        # plt.show()
        # plt.close(fig)

        if (train_mode == "train_CL") or (train_mode == "train_CL_and_inference"):
            hh_plot = hh.cpu().detach().numpy()
            hh_plot = np.reshape(hh_plot, tuple([len_batch, len_augs,] + list(hh_plot.shape[1:])))
            colors = plot_utils.get_N_colors(hh_plot.shape[0], mpl.colormaps['prism'])
            fig, ax = plot_utils.simple_plot()
            for ii in range(hh_plot.shape[0]):
                ax.scatter(hh_plot[ii,:,0], hh_plot[ii,:,1], c=colors[ii], lw=1)
            fig.savefig(path_save_figs + "pseudo_" + save_aux_fig_name + ".png")
            plt.tight_layout()
            plt.show()
            plt.close(fig)

        if (train_mode == "train_inference_fully_supervised") or (train_mode == "train_inference_from_latents") or (train_mode == "train_CL_and_inference"):
            theta_true_plot = theta_true.cpu().detach().numpy()
            theta_pred_plot = theta_pred.cpu().detach().numpy()
            theta_true_plot = np.reshape(theta_true_plot, tuple([len_batch, len_augs,] + [theta_true_plot.shape[-1],]))
            theta_pred_plot = np.reshape(theta_pred_plot, tuple([len_batch, len_augs,] + [theta_pred_plot.shape[-1],]))
            
            if inference_loss == "MSE":
                colors = plot_utils.get_N_colors(len_batch, mpl.colormaps['prism'])
                fig, axs = plt.subplots(1, theta_pred_plot.shape[-1], figsize=(5.2*theta_pred_plot.shape[-1], 5))
                for ii in range(len_batch):
                    for ii_cosmo_param in range(theta_pred_plot.shape[-1]):
                        ax = axs[ii_cosmo_param]            
                        ax.scatter(
                            theta_true_plot[ii, :, ii_cosmo_param], theta_pred_plot[ii, :, ii_cosmo_param],
                            color=colors[ii], alpha=1., marker ='o', s=6
                        )
                        ymax = np.nanmax(theta_true_plot[...,ii_cosmo_param])
                        ymin = np.nanmin(theta_true_plot[...,ii_cosmo_param])
                        tmp_xx = np.linspace(ymin, ymax, 2)
                        ax.plot(tmp_xx, tmp_xx, c='k', lw=2, ls='-', alpha=1)
                        ax.set_xlabel(r'True')
                    axs[0].set_ylabel(r'Pred')
                plt.tight_layout()
                plt.show()
                fig.savefig(path_save_figs + "inference_" + save_aux_fig_name + ".png")
                plt.close(fig)
                
            if inference_loss == "MultivariateNormal":
                Cov_plot = Cov.cpu().detach().numpy()
                Cov_plot = np.reshape(Cov_plot, tuple([len_batch, len_augs,] + list(Cov.shape[1:])))
                
                colors = plot_utils.get_N_colors(len_batch, mpl.colormaps['prism'])
                fig, axs = plt.subplots(1, theta_pred_plot.shape[-1], figsize=(5.2*theta_pred_plot.shape[-1], 5))
                for ii in range(len_batch):
                    for ii_cosmo_param in range(theta_pred_plot.shape[-1]):
                        ax = axs[ii_cosmo_param]            
                        ax.scatter(
                            theta_true_plot[ii, :, ii_cosmo_param], theta_pred_plot[ii, :, ii_cosmo_param],
                           color=colors[ii], alpha=1., marker ='o', s=6
                        )
                        ax.errorbar(
                            theta_true_plot[ii, :, ii_cosmo_param], theta_pred_plot[ii, :, ii_cosmo_param],
                            yerr=np.sqrt(Cov_plot[ii, :, ii_cosmo_param, ii_cosmo_param]),
                            c=colors[ii], ls='', capsize=2, alpha=0.6, elinewidth=1
                        )
                        ymax = np.nanmax(theta_true_plot[...,ii_cosmo_param])
                        ymin = np.nanmin(theta_true_plot[...,ii_cosmo_param])
                        tmp_xx = np.linspace(ymin, ymax, 2)
                        ax.plot(tmp_xx, tmp_xx, c='k', lw=2, ls='-', alpha=1)
                        ax.set_xlabel(r'True')
                    axs[0].set_ylabel(r'Pred')
                plt.tight_layout()
                plt.show()
                fig.savefig(path_save_figs + "inference_" + save_aux_fig_name + ".png")
                plt.close(fig)
                
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