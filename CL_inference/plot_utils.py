import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def matplotlib_default_config():

    font = {
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 10
    }
    
    rcnew = {
        "mathtext.fontset" : "cm", 
        "text.usetex": False,

        'figure.frameon': True,
        'axes.linewidth': 2.,

        "axes.titlesize" : 32, 
        "axes.labelsize" : 28,
        "legend.fontsize" : 28,
        'legend.fancybox': True,
        'lines.linewidth': 2.5,

        'xtick.alignment': 'center',
        'xtick.bottom': True,
        'xtick.color': 'black',
        'xtick.direction': 'in',
        'xtick.labelbottom': True,
        'xtick.labelsize': 24, #17.5,
        'xtick.labeltop': False,
        'xtick.major.bottom': True,
        'xtick.major.pad': 6.0,
        'xtick.major.size': 14.0,
        'xtick.major.top': True,
        'xtick.major.width': 1.5,
        'xtick.minor.bottom': True,
        'xtick.minor.pad': 3.4,
        'xtick.minor.size': 7.0,
        'xtick.minor.top': True,
        'xtick.minor.visible': True,
        'xtick.minor.width': 1.0,
        'xtick.top': True,

        'ytick.alignment': 'center_baseline',
        'ytick.color': 'black',
        'ytick.direction': 'in',
        'ytick.labelleft': True,
        'ytick.labelright': False,
        'ytick.labelsize': 24, #17.5,
        'ytick.left': True,
        'ytick.major.left': True,
        'ytick.major.pad': 6.0,
        'ytick.major.right': True,
        'ytick.major.size': 14.0,
        'ytick.major.width': 1.5,
        'ytick.minor.left': True,
        'ytick.minor.pad': 3.4,
        'ytick.minor.right': True,
        'ytick.minor.size': 7.0,
        'ytick.minor.visible': True,
        'ytick.minor.width': 1.0,
        'ytick.right': True
    }
    
    return font, rcnew


def get_N_colors(N, colormap):
    def get_colors(inp, colormap, vmin=None, vmax=None):
        norm = plt.Normalize(vmin, vmax)
        return colormap(norm(inp))
    colors = get_colors(np.linspace(1, N, num=N, endpoint=True), colormap)
    return colors


def get_N_markers(N):
    all_markers= np.array(['o','x','s','v','*','P','^','<','>','1','2','3','4','8','p','h','H','+','D','d','|','_','X',0,1,2,3,4,5,6,7,8,9,10,11])
    markers = all_markers[np.arange(N)]
    return markers


def get_N_linestyles(N):
    all_markers= [
        (0, ()),
        (0, (5, 2)),
        (0, (1, 1)),
        (0, (3, 1, 1, 1)),
        (0, (4, 2, 10, 2)),
        (0, (3, 5, 1, 5)),
        (0, (1, 10)),
        (0, (1, 1)),
        (5, (10, 3)),
        (0, (5, 10)),
        (0, (5, 1)),
        (0, (3, 10, 1, 10)),
        (0, (3, 5, 1, 5, 1, 5)),
        (0, (3, 10, 1, 10, 1, 10)),
        (0, (3, 1, 1, 1, 1, 1))
    ]
    linestyles = all_markers[:N]
    return linestyles


def simple_plot(x_label='x', y_label='y', custom_labels=None, custom_lines=None):
    
    fig, ax = plt.subplots(1,1, figsize=(7,5))
    fontsize = 24
    fontsize1 = 18
    
    ax.tick_params('both', length=5, width=2, which='major')
    [kk.set_linewidth(2) for kk in ax.spines.values()]
    
    ax.set_ylabel(y_label, size=fontsize)
    ax.yaxis.set_major_locator(mpl.pyplot.MaxNLocator(5))
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
    # for tick in ax.yaxis.get_major_ticks():
        # tick.label.set_fontsize(fontsize1)
    
    ax.set_xlabel(x_label, size=fontsize)
    ax.xaxis.set_major_locator(mpl.pyplot.MaxNLocator(5))
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    # for tick in ax.xaxis.get_major_ticks():
        # tick.label.set_fontsize(fontsize1)

    if custom_labels != None:
        legend = ax.legend(custom_lines, custom_labels, loc='upper right',
                           fancybox=True, shadow=True, ncol=1,fontsize=fontsize1)
        ax.add_artist(legend)

    return fig, ax


def corner_plot(theta, inferred_theta, custom_titles, dict_bounds=None, color_infer='crimson', fontsize=20, fontsize1=13, N_ticks=3, nbins_contour=30):

    fig, axs = plt.subplots(len(custom_titles), len(custom_titles), figsize=(2*len(custom_titles), 2*len(custom_titles)))

    for ii in range(len(custom_titles)):

        for jj in range(ii+1, len(custom_titles)):
            axs[ii, jj].set_axis_off()

        ax = axs[ii, ii]
        
        [kk.set_linewidth(1.5) for kk in ax.spines.values()]
        
        if dict_bounds != None:
            ax.axvline(dict_bounds[list(dict_bounds.keys())[ii]][0], color='fuchsia', lw=2, ls='-')
            ax.axvline(dict_bounds[list(dict_bounds.keys())[ii]][1], color='fuchsia', lw=2, ls='-')
        
        tmp_xx = inferred_theta[:, ii]
        tmp_xx_true = theta[ii]

    #     min_x, max_x = dict_bounds[key][0], dict_bounds[key][1]
        min_x, max_x = np.min(tmp_xx), np.max(tmp_xx)

        tmp_hist = tmp_xx
        counts, bin_edges = np.histogram(tmp_hist, bins=20, range=(min_x, max_x))
        bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
        tmp_y_hist = np.array(counts/np.sum(counts))
        ax.plot(bin_centers, tmp_y_hist, color=color_infer, lw=2)

        tmp_mean = tmp_xx.mean()
        tmp_std = tmp_xx.std()
        ax.axvspan(tmp_mean-tmp_std, tmp_mean+tmp_std, alpha=0.2, color=color_infer)
        ax.axvline(tmp_mean+tmp_std, color=color_infer, ls="--", lw=1)
        ax.axvline(tmp_mean-tmp_std, color=color_infer, ls="--", lw=1)
        ax.axvline(tmp_xx_true, color='k', lw=1.2, ls='--')
        
        tmp_custom_ticks = np.linspace(min_x, max_x, N_ticks+2)[1:-1]
        ax.set_xticks(tmp_custom_ticks)
        
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        tmp_custom_ticks = np.linspace(0, np.max(tmp_y_hist), N_ticks)
        tmp_custom_labels = np.around(tmp_custom_ticks, 2).astype(str)
        ax.set_yticks(tmp_custom_ticks)
        ax.set_yticklabels(tmp_custom_labels, fontsize=fontsize1, color='k', rotation=0, verticalalignment='bottom')
        ax.set_ylabel(r'$P ($' +custom_titles[ii] + r'$)$', fontsize=fontsize, rotation=-60, labelpad=35)
        
        for jj in range(ii+1, len(custom_titles)):

            ax = axs[jj, ii]
            
            [kk.set_linewidth(1.5) for kk in ax.spines.values()]
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            tmp_yy = inferred_theta[:, jj]
            tmp_yy_true = theta[jj]

            ax.axvline(tmp_xx_true, color='k', lw=1.2, ls='--')
            ax.axhline(tmp_yy_true, color='k', lw=1.2, ls='--')
            
            if dict_bounds != None:
                ax.axvline(dict_bounds[list(dict_bounds.keys())[ii]][0], color='fuchsia', lw=2, ls='-')
                ax.axvline(dict_bounds[list(dict_bounds.keys())[ii]][1], color='fuchsia', lw=2, ls='-')
                ax.axhline(dict_bounds[list(dict_bounds.keys())[jj]][0], color='fuchsia', lw=2, ls='-')
                ax.axhline(dict_bounds[list(dict_bounds.keys())[jj]][1], color='fuchsia', lw=2, ls='-')
                # min_y, max_y = dict_bounds[list(dict_bounds.keys())[jj]][0], dict_bounds[list(dict_bounds.keys())[jj]][1]
            
            min_y, max_y = np.min(tmp_yy), np.max(tmp_yy)

            xi, yi = np.mgrid[
                tmp_xx.min():tmp_xx.max():nbins_contour*1j,
                tmp_yy.min():tmp_yy.max():nbins_contour*1j
            ]
            values = np.vstack([tmp_xx, tmp_yy])
            kernel = sp.stats.gaussian_kde(values)
            positions = np.vstack([xi.ravel(), yi.ravel()])
            zi = np.reshape(kernel(positions).T, xi.shape)
            zi = zi / zi.sum()

            t = np.linspace(0, zi.max(), 1000)
            integral = ((zi >= t[:, None, None]) * zi).sum(axis=(1,2))
            f = sp.interpolate.interp1d(integral, t)
            t_contours = f(np.array([1-0.68]))

            pc = ax.pcolormesh(
                xi, yi,
                zi.reshape(xi.shape),
                shading='gouraud', cmap=mpl.colormaps['coolwarm']
            )

            ax.contour(
                zi.T,
                t_contours,
                extent=[tmp_xx.min(),tmp_xx.max(),tmp_yy.min(),tmp_yy.max()],
                colors=[color_infer],
                linewidths=1,
                linestyles='--'
            )

            ax.scatter(tmp_xx, tmp_yy, s=.01, c=color_infer, alpha=0.5)
            ax.scatter(tmp_xx_true, tmp_yy_true, s=30, c='k', marker='x')

            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])
            if ii == 0:
                tmp_custom_ticks = np.linspace(min_y, max_y, N_ticks+2)[1:-1]
                tmp_custom_labels = np.around(tmp_custom_ticks, 3).astype(str)
                ax.set_yticks(tmp_custom_ticks)
                ax.set_yticklabels(tmp_custom_labels, fontsize=fontsize1, color='k', rotation=60)
                ax.set_ylabel(custom_titles[jj], fontsize=fontsize)
            else:
                tmp_custom_ticks = np.linspace(min_y, max_y, N_ticks+2)[1:-1]
                ax.set_yticks(tmp_custom_ticks)
                tmp_custom_ticks = np.linspace(min_x, max_x, N_ticks+2)[1:-1]
                ax.set_xticks(tmp_custom_ticks)

        tmp_custom_ticks = np.linspace(min_x, max_x, N_ticks+2)[1:-1]
        tmp_custom_labels = np.around(tmp_custom_ticks, 3).astype(str)
        axs[-1,ii].set_xticks(tmp_custom_ticks)
        axs[-1,ii].set_xticklabels(tmp_custom_labels, fontsize=fontsize1, color='k', rotation=30)
        axs[-1,ii].set_xlabel(custom_titles[ii], fontsize=fontsize)
        axs[ii,ii].set_xlim([min_x, max_x])
        
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.07, wspace=0.07)
    
    return fig, axs
    
    
def plot_inference_split_models(
    list_model_names,
    len_models,
    theta_true,
    theta_pred,
    Cov,
    custom_titles,
    limits_plots,
    fontsize=26,
    fontsize1=18,
    alpha=0.1,
    colors=None
):

    fig, axs = plt.subplots(
        len(list_model_names), theta_pred.shape[-1], figsize=(5.2*theta_pred.shape[-1], 5.2*len(list_model_names))
    )

    ii_aug_column0 = 0
    if colors == None:
        colors = get_N_colors(len(list_model_names), mpl.colormaps['prism'])
    for ii_model, model_name in enumerate(list_model_names):    
        for ii_cosmo_param in range(theta_pred.shape[-1]):
            if len(list_model_names)==1:
                ax = axs[ii_cosmo_param]
            else:
                ax = axs[ii_model, ii_cosmo_param]
            if ii_model == 0:
                ax.set_title(custom_titles[ii_cosmo_param], size=fontsize+8, pad=16)
            if ii_model == len(list_model_names)-1:
                ax.set_xlabel(r'True ', size=fontsize)
            else:
                ax.axes.get_xaxis().set_visible(False)

            for ii_aug in range(len_models[ii_model]):
                ii_column_aug = ii_aug_column0 + ii_aug

                ax.scatter(
                    theta_true[:, ii_column_aug, ii_cosmo_param], theta_pred[:, ii_column_aug, ii_cosmo_param],
                   color=colors[ii_model], marker ='o', s=3, alpha=alpha
                )
                ax.errorbar(
                    theta_true[:, ii_column_aug, ii_cosmo_param], theta_pred[:, ii_column_aug, ii_cosmo_param],
                    yerr=np.sqrt(Cov[:, ii_column_aug, ii_cosmo_param, ii_cosmo_param]),
                    c=colors[ii_model], ls='', capsize=2, alpha=alpha, elinewidth=1
                )

            ymin = limits_plots[ii_cosmo_param][0]
            ymax = limits_plots[ii_cosmo_param][1]
    #     ymax = np.nanmax(theta_true[..., ii_cosmo_param])
    #     ymin = np.nanmin(theta_true[..., ii_cosmo_param])
            tmp_xx = np.linspace(ymin, ymax, 2)
            ax.plot(tmp_xx, tmp_xx, c='k', lw=2, ls='-', alpha=1)
            ax.set_xlim([ymin, ymax])
            ax.set_ylim([ymin, ymax])

        ii_aug_column0 += len_models[ii_model]

        custom_lines = []
        custom_labels = []
        custom_lines.append(mpl.lines.Line2D([0], [0], color=colors[ii_model], ls='-', lw=10, marker=None, markersize=9))
        custom_labels.append(model_name)
        if len(list_model_names)==1:
            ax = axs[0]
        else:
            ax = axs[ii_model, 0]
        legend = ax.legend(custom_lines,custom_labels,loc='upper left',fancybox=True, shadow=True, ncol=1,fontsize=16)
        ax.set_ylabel(r'Pred ', size=fontsize)
            
    return fig, axs
    
    
def theta_distrib_plot(dsets, custom_titles, fontsize=20, fontsize1=13, N_ticks=3, colors=['limegreen', 'royalblue', 'red', 'k']):

    min_theta = []
    max_theta = []
    custom_lines = []
    custom_labels = []
    for ii, key in enumerate(dsets.keys()):
        min_theta.append(np.min(dsets[key].theta, axis=0))
        max_theta.append(np.max(dsets[key].theta, axis=0))
        custom_lines.append(mpl.lines.Line2D([0], [0], color=colors[ii], ls='-', lw=10, marker=None, markersize=9))
        custom_labels.append(key)
    min_theta = np.array(min_theta)
    max_theta = np.array(max_theta)

    NN_theta = dsets[key].theta.shape[-1]

    fig, axs = plt.subplots(NN_theta, NN_theta, figsize=(2*NN_theta, 2*NN_theta))
    for ii in range(NN_theta):

        for jj in range(ii+1, NN_theta):
            axs[ii, jj].set_axis_off()

        ax = axs[ii, ii]

        min_x = np.min(min_theta, axis=0)[ii]
        max_x = np.max(max_theta, axis=0)[ii]

        [kk.set_linewidth(1.5) for kk in ax.spines.values()]

        ax.set_xticklabels([])

        for kk, key in enumerate(dsets.keys()):
            ax.axvline(min_theta[kk][ii], color=colors[kk], lw=1, ls='-', alpha=1)
            ax.axvline(max_theta[kk][ii], color=colors[kk], lw=1, ls='-', alpha=1)

            tmp_hist = dsets[key].theta[:, ii]
            counts, bin_edges = np.histogram(tmp_hist, bins=20, range=(min_x, max_x))
            bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
            tmp_y_hist = np.array(counts)
            ax.plot(bin_centers, tmp_y_hist, color=colors[kk], lw=2)

        tmp_custom_ticks = np.linspace(min_x, max_x, N_ticks+2)[1:-1]
        ax.set_xticks(tmp_custom_ticks)

        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        tmp_custom_ticks = np.linspace(0, np.max(tmp_y_hist), N_ticks)
        tmp_custom_labels = np.around(tmp_custom_ticks, 2).astype(str)
        ax.set_yticks(tmp_custom_ticks)
        ax.set_yticklabels(tmp_custom_labels, fontsize=fontsize1, color='k', rotation=0, verticalalignment='bottom')
        ax.set_ylabel(r'$\# $' +custom_titles[ii], fontsize=fontsize, rotation=-60, labelpad=35)

        for jj in range(ii+1, NN_theta):

            ax = axs[jj, ii]

            [kk.set_linewidth(1.5) for kk in ax.spines.values()]
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            for kk, key in enumerate(dsets.keys()):
                ax.axvline(min_theta[kk][ii], color=colors[kk], lw=1, ls='-', alpha=1)
                ax.axvline(max_theta[kk][ii], color=colors[kk], lw=1, ls='-', alpha=1)
                ax.axhline(min_theta[kk][jj], color=colors[kk], lw=1, ls='-', alpha=1)
                ax.axhline(max_theta[kk][jj], color=colors[kk], lw=1, ls='-', alpha=1)
                ax.scatter(dsets[key].theta[:, ii], dsets[key].theta[:, jj], color=colors[kk], s=.1, alpha=0.6)

            min_x = np.min(min_theta, axis=0)[ii]
            max_x = np.max(max_theta, axis=0)[ii]
            min_y = np.min(min_theta, axis=0)[jj]
            max_y = np.max(max_theta, axis=0)[jj]
            ax.set_xlim([min_x-0.06*min_x, max_x+0.06*min_x])
            ax.set_ylim([min_y-0.06*min_y, max_y+0.06*min_y])
            if ii == 0:
                tmp_custom_ticks = np.linspace(min_y, max_y, N_ticks+2)[1:-1]
                tmp_custom_labels = np.around(tmp_custom_ticks, 3).astype(str)
                ax.set_yticks(tmp_custom_ticks)
                ax.set_yticklabels(tmp_custom_labels, fontsize=fontsize1, color='k', rotation=60)
                ax.set_ylabel(custom_titles[jj], fontsize=fontsize)
            else:
                tmp_custom_ticks = np.linspace(min_y, max_y, N_ticks+2)[1:-1]
                ax.set_yticks(tmp_custom_ticks)
                tmp_custom_ticks = np.linspace(min_x, max_x, N_ticks+2)[1:-1]
                ax.set_xticks(tmp_custom_ticks)

        tmp_custom_ticks = np.linspace(min_x, max_x, N_ticks+2)[1:-1]
        tmp_custom_labels = np.around(tmp_custom_ticks, 3).astype(str)
        axs[-1,ii].set_xticks(tmp_custom_ticks)
        axs[-1,ii].set_xticklabels(tmp_custom_labels, fontsize=fontsize1, color='k', rotation=30)
        axs[-1,ii].set_xlabel(custom_titles[ii], fontsize=fontsize)
        axs[ii,ii].set_xlim([min_x-0.06*min_x, max_x+0.06*min_x])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.07, wspace=0.07)

    legend = fig.legend(custom_lines, custom_labels, loc='upper right',
                       fancybox=True, shadow=True, ncol=1,fontsize=16)
    fig.add_artist(legend)

    return fig, axs
    
    
def plot_xx_from_sampled_posteriors(xx, inferred_xx, kk,
                                   fontsize=24, fontsize1=18, res_y_lims=[0.998, 1.002],
                                   y_label=r'$P(k) \left[ \left(h^{-1} \mathrm{Mpc}\right)^{3} \right]$',
                                   x_label=r'Wavenumber $k \left[ h\, \mathrm{Mpc}^{-1} \right]$',
                                   y_label_res=r'${P_{\mathrm{infer}}}/{P_{\mathrm{true}}}$'):
    
    fig = plt.figure(figsize=(7, 6))

    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    ax = plt.subplot(gs[0])

    ax.yaxis.set_major_locator(mpl.pyplot.MaxNLocator(5))
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
    # for tick in ax.yaxis.get_major_ticks():
        # tick.label.set_fontsize(fontsize1) 
    ax.set_xticks([])

    ax.tick_params('both', length=5, width=2, which='major')
    [kk.set_linewidth(2) for kk in ax.spines.values()]

    ax_res = plt.subplot(gs[1])
    ax_res.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    ax_res.xaxis.set_major_locator(mpl.pyplot.MaxNLocator(5))
    # for tick in ax_res.xaxis.get_major_ticks():
        # tick.label.set_fontsize(fontsize1)

    tmp_custom_ticks = np.linspace(res_y_lims[0], res_y_lims[1], 3)
    tmp_custom_labels = np.around(tmp_custom_ticks, 4).astype(str)
    ax_res.set_yticks(tmp_custom_ticks)
    ax_res.set_yticklabels(tmp_custom_labels, fontsize=fontsize1, color='k', rotation=60)

    ax_res.tick_params('both', length=5, width=2, which='major')
    [kk.set_linewidth(2) for kk in ax_res.spines.values()]

    ax_res.set_ylim([res_y_lims[0], res_y_lims[1]])
    ax.set_ylabel(y_label, size=fontsize)
    ax_res.set_ylabel(y_label_res, size=fontsize)
    ax_res.set_xlabel(x_label, size=fontsize)

    colors = get_N_colors(xx.shape[0], mpl.colormaps['prism'])
    custom_lines = []
    custom_labels = []
    for ii_sample in range(xx.shape[0]):

        tmp_inferred_xx = inferred_xx[ii_sample]
        tmp_xx = xx[ii_sample]

        tmp_xx_plot = tmp_inferred_xx.T
        ax.plot(np.log10(kk), tmp_xx_plot, c=colors[ii_sample], alpha=0.1, lw=0.2)
        ax.plot(np.log10(kk), tmp_xx, c=colors[ii_sample], ls=':', lw=4, alpha=0.8)

        ax_res.plot(np.log10(kk), (tmp_xx_plot.T / tmp_xx).T, c=colors[ii_sample], alpha=0.1, lw=0.1)

        custom_lines.append(mpl.lines.Line2D([0], [0], color=colors[ii_sample], ls='-', lw=0, marker='s', markersize=16))
        custom_labels.append('# ' + str(ii_sample))
    legend = ax.legend(custom_lines, custom_labels, loc='upper right',
                       fancybox=True, shadow=True, ncol=1,fontsize=fontsize1)
    ax.add_artist(legend)    

    custom_lines = [
        mpl.lines.Line2D([0], [0], color='grey', ls=':', lw=4, marker=None, markersize=9),
        mpl.lines.Line2D([0], [0], color='grey', ls='-', lw=1, marker=None, markersize=9)
    ]
    custom_labels = [
        'True',
        'Posterior samples'
    ]
    legend = ax.legend(custom_lines, custom_labels, loc='lower left',
                       fancybox=True, shadow=True, ncol=1,fontsize=fontsize1)
    ax.add_artist(legend)    

    ax_res.axhline(1, ls=':', lw=4, c='k')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.04)
    
    return fig, ax, ax_res
    
    
def plot_rank_statistcis(ranks, N_sampled_ranks, custom_titles=[r'$\Omega_\mathrm{c}$', r'$\Omega_\mathrm{b}$', r'$h$', r'$n_\mathrm{s}$', r'$\sigma_{8,\mathrm{c}}$'], N_bins=None, fontsize=22, fontsize1=18):
    
    if N_bins == None: N_bins=int(N_sampled_ranks/50)
    
    fig, axs = plt.subplots(1, len(custom_titles), figsize=(3.5*len(custom_titles), 4), sharex=True, sharey=True)
    
    axs[0].set_ylabel(r'$P(\mathrm{Rank\, Counts})$', size=fontsize)
    
    tmp_max = 0
    tmp_min = 1
    for ii in range(len(custom_titles)):
        ax = axs[ii]

        ax.tick_params('both', length=5, width=2, which='major')
        [kk.set_linewidth(2) for kk in ax.spines.values()]
        tmp_custom_ticks = np.linspace(0, 1, 5)[1:-1]
        tmp_custom_labels = np.around(tmp_custom_ticks, 3).astype(str)
        ax.set_xticks(tmp_custom_ticks)
        ax.set_xticklabels(tmp_custom_labels, fontsize=fontsize1, color='k', rotation=0)
        ax.set_xlabel(r'Rank (normalized)', size=fontsize)

        ax.axhline(1/N_bins, color='k', lw=2, ls=':')

        tmp_hist = ranks[:, ii]/N_sampled_ranks
        counts, bin_edges = np.histogram(tmp_hist, bins=N_bins, range=(0, 1))
        bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
        tmp_y_hist = np.array(counts) / ranks.shape[0]
        ax.plot(bin_centers, tmp_y_hist, color='royalblue', lw=2)    
        
        tmp_max = np.max([tmp_max, np.max(tmp_y_hist)])
        tmp_min = np.min([tmp_min, np.min(tmp_y_hist)])
        
        ax.set_title(custom_titles[ii], fontsize=fontsize)
        ax.set_xlim([0, 1])

    tmp_custom_ticks = np.linspace(tmp_min, tmp_max, 5)[1:-1]
    tmp_custom_labels = np.around(tmp_custom_ticks, 3).astype(str)
    axs[0].set_yticks(tmp_custom_ticks)
    axs[0].set_yticklabels(tmp_custom_labels, fontsize=fontsize1, color='k', rotation=0)
        
    ax.set_ylim([tmp_min, tmp_max])
    
    custom_lines = [
        mpl.lines.Line2D([0], [0], color='royalblue', ls='-', lw=2, marker=None, markersize=9)
    ]
    custom_labels = [
        'Model Rank'
    ]
    legend = axs[0].legend(custom_lines, custom_labels, loc='upper right',
                       fancybox=True, shadow=True, ncol=1,fontsize=fontsize1)
    axs[0].add_artist(legend)  

    custom_lines = [
        mpl.lines.Line2D([0], [0], color='k', ls=':', lw=2, marker=None, markersize=9)
    ]
    custom_labels = [
        'Expected'
    ]
    legend = axs[1].legend(custom_lines, custom_labels, loc='upper right',
                       fancybox=True, shadow=True, ncol=1,fontsize=fontsize1)
    axs[1].add_artist(legend)      
    
    return fig, axs
    
    
def plot_parameter_prediction_vs_truth(inferred_theta, theta_true, custom_titles=[r'$\Omega_\mathrm{c}$', r'$\Omega_\mathrm{b}$', r'$h$', r'$n_\mathrm{s}$', r'$\sigma_{8,\mathrm{c}}$'], fontsize=22, fontsize1=18):
    
    fig, axs = plt.subplots(1, len(custom_titles), figsize=(4.2*len(custom_titles), 4))
    
    means = np.mean(inferred_theta, axis=1)
    stds = np.std(inferred_theta, axis=1)
    for ii in range(len(custom_titles)):
        ax = axs[ii]

        tmp_means = means[:, ii]
        tmp_stds = stds[:, ii]
        ax.errorbar(
            theta_true[:, ii],
            tmp_means,
            tmp_stds,
            color='royalblue', alpha=0.3,
            fmt ='s', ms=2, ls=None,
            elinewidth=0.5
        )

        tmp_xx = np.linspace(np.min(theta_true[:, ii]), np.max(theta_true[:, ii]), 2)
        ax.plot(tmp_xx, tmp_xx, c='k', lw=2, ls='-', alpha=1)

        ax.tick_params('both', length=5, width=2, which='major')
        [kk.set_linewidth(2) for kk in ax.spines.values()]
        
        ax.set_xlim([np.min(theta_true[:, ii]), np.max(theta_true[:, ii])])
        tmp_custom_ticks = np.linspace(np.min(theta_true[:, ii]), np.max(theta_true[:, ii]), 3)
        tmp_custom_labels = np.around(tmp_custom_ticks, 3).astype(str)
        ax.set_xticks(tmp_custom_ticks)
        ax.set_xticklabels(tmp_custom_labels, fontsize=fontsize1, color='k', rotation=0)
        
        ax.set_ylim([np.min(tmp_means-tmp_stds), np.max(tmp_means+tmp_stds)])    
        tmp_custom_ticks = np.linspace(np.min(tmp_means-tmp_stds), np.max(tmp_means+tmp_stds), 3)
        tmp_custom_labels = np.around(tmp_custom_ticks, 3).astype(str)
        ax.set_yticks(tmp_custom_ticks)
        ax.set_yticklabels(tmp_custom_labels, fontsize=fontsize1, color='k', rotation=0)
        
        ax.set_title(custom_titles[ii], fontsize=fontsize)
        ax.set_xlabel(r'True '+ custom_titles[ii], size=fontsize1)
        ax.set_ylabel(r'Pred '+ custom_titles[ii], size=fontsize1)
            
    return fig, axs
    
    
def plot_parameter_regression_vs_truth(theta_true, theta_pred, custom_titles=[r'$\Omega_\mathrm{c}$', r'$\Omega_\mathrm{b}$', r'$h$', r'$n_\mathrm{s}$', r'$\sigma_{8,\mathrm{c}}$'], fontsize=22, fontsize1=18, cmap='jet'):
  
    if (len(theta_pred.shape)==3):
        theta_true = theta_true[:,np.newaxis].repeat(theta_pred.shape[1], axis=1)
    if (len(theta_pred.shape)==2):
        theta_pred = theta_pred[:,np.newaxis]
        theta_true = theta_true[:,np.newaxis]
    
    fig, axs = plt.subplots(1, len(custom_titles), figsize=(4.2*len(custom_titles), 4))
    
    colors = get_N_colors(theta_pred.shape[1], mpl.colormaps[cmap])
    for ii_cosmo_param in range(len(custom_titles)):
        ax = axs[ii_cosmo_param]
        
        for ii_aug in range(theta_pred.shape[1]):
            ax.scatter(theta_true[:, ii_aug, ii_cosmo_param], theta_pred[:, ii_aug, ii_cosmo_param], color=colors[ii_aug], alpha=0.3, marker ='s', s=2)

        tmp_xx = np.linspace(np.min(theta_true[:, :, ii_cosmo_param]), np.max(theta_true[:, :, ii_cosmo_param]), 2)
        ax.plot(tmp_xx, tmp_xx, c='k', lw=2, ls='-', alpha=1)

        ax.tick_params('both', length=5, width=2, which='major')
        [kk.set_linewidth(2) for kk in ax.spines.values()]

        ax.set_xlim([np.min(theta_true[:, :, ii_cosmo_param]), np.max(theta_true[:, :, ii_cosmo_param])])
        tmp_custom_ticks = np.linspace(np.min(theta_true[:, :, ii_cosmo_param]), np.max(theta_true[:, :, ii_cosmo_param]), 3)
        tmp_custom_labels = np.around(tmp_custom_ticks, 3).astype(str)
        ax.set_xticks(tmp_custom_ticks)
        ax.set_xticklabels(tmp_custom_labels, fontsize=fontsize1, color='k', rotation=0)

        ax.set_ylim([np.min(theta_pred[:, :, ii_cosmo_param]), np.max(theta_pred[:, :, ii_cosmo_param])])    
        tmp_custom_ticks = np.linspace(np.min(theta_pred[:, :, ii_cosmo_param]), np.max(theta_pred[:, :, ii_cosmo_param]), 3)
        tmp_custom_labels = np.around(tmp_custom_ticks, 3).astype(str)
        ax.set_yticks(tmp_custom_ticks)
        ax.set_yticklabels(tmp_custom_labels, fontsize=fontsize1, color='k', rotation=0)

        ax.set_title(custom_titles[ii_cosmo_param], fontsize=fontsize)
        ax.set_xlabel(r'True '+ custom_titles[ii_cosmo_param], size=fontsize1)
        ax.set_ylabel(r'Pred '+ custom_titles[ii_cosmo_param], size=fontsize1)
                
    return fig, axs
    
    
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


def compute_err_hist_augs(Cov, min_x=[0,0,0,0,0,0,0,0,0,0,0,0], max_x=[0.05, 0.012, 0.12, 0.042, 0.06, 3.2, 1.5, 1.5, 3., .4, 1., 3.], bins=60):
    
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