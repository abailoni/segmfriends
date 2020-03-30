import matplotlib
matplotlib.use('Agg')
import os
from matplotlib import pyplot as plt
import numpy as np

from ..transform.segm_to_bound import compute_boundary_mask_from_label_image
# from segmfriends.transform.inferno.temp_crap import FindBestAgglFromOversegmAndGT
# from segmfriends.features import from_affinities_to_hmap


# Build random color map:
__MAX_LABEL__ = 100000000
rand_cm = matplotlib.colors.ListedColormap(np.random.rand(__MAX_LABEL__, 3))


matplotlib.rcParams.update({'font.size': 5})
# DEF_INTERP = 'none'
DEF_INTERP = 'nearest'
segm_plot_kwargs = {'vmax': __MAX_LABEL__, 'vmin':0}


def mask_the_mask(mask, value_to_mask=0., interval=None):
    if interval is not None:
        return np.ma.masked_where(np.logical_and(mask < interval[1], mask > interval[0]), mask)
    else:
        return np.ma.masked_where(np.logical_and(mask < value_to_mask+1e-3, mask > value_to_mask-1e-3), mask)

def mask_array(array_to_mask, mask):
    return np.ma.masked_where(mask, array_to_mask)

def get_bound_mask(segm):
    # print("B mask is expensive...")
    return compute_boundary_mask_from_label_image(segm,
                                                  np.array([[0,1,0], [0,0,1]]),
                                                  compress_channels=True)

def get_masked_boundary_mask(segm):
    #     bound = np.logical_or(get_boundary_mask(segm)[0, 0],get_boundary_mask(segm)[1, 0])
    bound = get_bound_mask(segm)
    return mask_the_mask(bound)

def plot_segm(target, segm, z_slice=0, background=None, mask_value=None, highlight_boundaries=True, plot_label_colors=True,
              alpha_labels=0.4, alpha_boundary=0.5, cmap=None):
    """Shape of expected background: (z,x,y)"""
    if background is not None:
        target.matshow(background[z_slice], cmap='gray', interpolation=DEF_INTERP)

    if mask_value is not None:
        segm = mask_the_mask(segm,value_to_mask=mask_value)
    if plot_label_colors:
        cmap = rand_cm if cmap is None else cmap
        target.matshow(segm[z_slice], cmap=cmap, alpha=alpha_labels, interpolation=DEF_INTERP, **segm_plot_kwargs)
    masked_bound = get_masked_boundary_mask(segm)
    if highlight_boundaries:
        target.matshow(masked_bound[z_slice], cmap='gray', alpha=alpha_boundary, interpolation=DEF_INTERP)
    return masked_bound

def find_splits_merges(target, GT_labels, segm, z_slice=0, background=None):
    GT_bound = get_bound_mask(GT_labels) * 3.
    segm_bound = get_bound_mask(segm) * (1.)
    diff_bound = (GT_bound+segm_bound).astype(np.int32)

    if background is not None:
        target.matshow(background[z_slice], cmap='gray', interpolation=DEF_INTERP)
    target.matshow(mask_the_mask(diff_bound==4)[z_slice], cmap='summer', alpha=1, interpolation=DEF_INTERP)
    target.matshow(mask_the_mask(diff_bound == 1)[z_slice], cmap='winter', alpha=1,
                interpolation=DEF_INTERP)
    target.matshow(mask_the_mask(diff_bound == 3)[z_slice], cmap='autumn', alpha=1,
                interpolation=DEF_INTERP)





def plot_output_affin(target, out_affin, nb_offset=1, z_slice=0):
    # Select the ones along x:
    cax = target.matshow(out_affin[nb_offset,z_slice,:,:], cmap=plt.get_cmap('seismic'), vmin=0, vmax=1, interpolation=DEF_INTERP)

def plot_gray_image(target, image, z_slice=0):
    # Select the ones along x:
    cax = target.matshow(image[z_slice, :, :], cmap='gray',
                         interpolation=DEF_INTERP)

def plot_affs_divergent_colors(ax, out_affin, type='pos', z_slice=0):
    # Select the ones along x:
    if type=='pos':
        data = out_affin * (out_affin>0. )
    else:
        data = out_affin * (out_affin < 0.)

    cax = ax.matshow(mask_the_mask(data[0,z_slice,:,:]), cmap=plt.get_cmap('autumn'), interpolation=DEF_INTERP)
    # cax = ax.matshow(mask_the_mask(neg_out_affin[axis, z_slice, :, :]), cmap=plt.get_cmap('cool'),
    #                  interpolation=DEF_INTERP)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def plot_lookahead(ax, lookahead, mergers=True, z_slice=0):
    # cax = ax.matshow(mask_the_mask(lookahead[z_slice, :, :, 1], value_to_mask=-2.0), cmap=plt.get_cmap('viridis'),
    #                  interpolation=DEF_INTERP)
    channel = 0 if mergers else 1
    cax = ax.matshow(mask_the_mask(lookahead[z_slice, :, :, channel], value_to_mask=-2.0), cmap=plt.get_cmap('jet'),
                     interpolation=DEF_INTERP)
    if mergers:
        mask_alive_boundaries = lookahead[z_slice, :, :, 1] > - 2.0
        cax = ax.matshow(mask_the_mask(mask_alive_boundaries),
                         cmap=plt.get_cmap('spring'),
                         interpolation=DEF_INTERP)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def get_figure(ncols, nrows, hide_axes=True, figsize=None):
    figsize = (ncols, nrows) if figsize is None else figsize
    f, ax = plt.subplots(ncols=ncols, nrows=nrows,
                         figsize=figsize)
    if hide_axes:
        for a in f.get_axes():
            a.axis('off')
    return f, ax

def save_plot(f, path, file_name):
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.tight_layout()
    if file_name.endswith('pdf'):
        f.savefig(os.path.join(path, file_name), format='pdf')
    # elif file_name.endswith('png'):
    #     f.savefig(os.path.join(path, file_name))
    else:
        f.savefig(os.path.join(path, file_name))
        # raise ValueError("File extension not supported atm")

def set_log_tics(ax, sub_range, sub_ticks, format = "%.2f", axis='x'):
    """
    :param sub_range:  [-2, 3]  # powers for the main range (from 0.01 to 1000)
    :param sub_ticks: [10, 11, 12, 14, 16, 18, 22, 25, 35, 45] put a tick every 10, every 12, 14...
    :param format: standard float string formatting
    :param axis: 'x', 'y'
    :return:
    """
    if axis == 'y':
        ax.set_yscale("log")

        # user controls
        #####################################################

        # set scalar and string format floats
        #####################################################
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(format))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter(format))

        # force 'autoscale'
        #####################################################
        yd = []  # matrix of y values from all lines on plot
        # cs = ax.collections[0]
        # cs.set_offset_position('data')
        # offs = cs.get_offsets()
        #
        # for n in range(len(offs)):
        #     yd.append(offs[n, 1])
        for n in range(len(plt.gca().get_lines())):
            line = plt.gca().get_lines()[n]
            yd.append((line.get_ydata()).tolist())
        yd = [item for sublist in yd for item in sublist]
        ymin, ymax = np.min(yd), np.max(yd)
        ax.set_ylim([0.9 * ymin, 1.1 * ymax])

        # add sub minor ticks
        #####################################################
        set_sub_formatter = []
        for i in sub_ticks:
            for j in range(sub_range[0], sub_range[1]):
                set_sub_formatter.append(i * 10 ** j)
        k = []
        for l in set_sub_formatter:
            if ymin < l < ymax:
                k.append(l)
        ax.set_yticks(k)
        ####################################################
    elif   axis == 'x':
        ax.set_xscale("log")

        # user controls
        #####################################################

        # set scalar and string format floats
        #####################################################
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(format))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter(format))

        # force 'autoscale'
        #####################################################
        yd = []  # matrix of y values from all lines on plot
        for n in range(len(plt.gca().get_lines())):
            line = plt.gca().get_lines()[n]
            yd.append((line.get_xdata()).tolist())
        yd = [item for sublist in yd for item in sublist]
        ymin, ymax = np.min(yd), np.max(yd)
        ax.set_xlim([0.9 * ymin, 1.1 * ymax])

        # add sub minor ticks
        #####################################################
        set_sub_formatter = []
        for i in sub_ticks:
            for j in range(sub_range[0], sub_range[1]):
                set_sub_formatter.append(i * 10 ** j)
        k = []
        for l in set_sub_formatter:
            if ymin < l < ymax:
                k.append(l)
        ax.set_xticks(k)
        ####################################################
    else:
        raise ValueError("Axis must be 'x' or 'y'" )
