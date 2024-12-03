import matplotlib as mpl
import matplotlib.axes as axes
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt

import numpy as np


cmap_big = mpl.colormaps['twilight_shifted'].resampled(int(1e3))
cmap_resc = mplcolors.ListedColormap(cmap_big(np.linspace(0.075, 0.925, 10000)))


def _create_plot(ax: axes.Axes, data: np.ndarray, k: int, max_value: float, axis: int, 
                 colmap: str|mplcolors.Colormap|None = None):
    cmap = colmap if colmap else cmap_resc
    match axis:
        case 1:
            d = data[k, :, :]
        case 2:
            d = data[:, k, :]
        case 3:
            d = data[:, :, k]
        case _:
            raise NotImplementedError("Axis not implemented")

    img = ax.imshow(d, cmap=cmap, vmax=max_value)
    ax.set_xticks([]) 
    ax.set_yticks([])
    if axis == 1:
        ax.set_xlabel(r"$k_2$", fontsize=14)
        ax.set_ylabel(r"$k_3$", fontsize=14)
    elif axis == 2:
        ax.set_xlabel(r"$k_1$", fontsize=14)
        ax.set_ylabel(r"$k_3$", fontsize=14)
    elif axis == 3:
        ax.set_xlabel(r"$k_1$", fontsize=14)
        ax.set_ylabel(r"$k_2$", fontsize=14)
    else:
        raise NotImplementedError("Axis not implemented")
    return img


def plot_section(data: np.ndarray, k: int, axis: int = 3, figsize: tuple[int, int] = (8,6), 
                 colmap: str|mplcolors.Colormap|None = None):
    max_value = np.max(data)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    img = _create_plot(ax, data, k, max_value, axis, colmap)
    fig.colorbar(img, ax=ax)


def plot_comparison(target: np.ndarray, pred: np.ndarray, k: int, figsize: tuple[int, int] = (14,6), axis: int = 3, 
                    colmap: str|mplcolors.Colormap|None = None, show_title: bool = True):
    max_value = np.max([target, pred])
    fig, axs = plt.subplots(1, 2, figsize=figsize, subplot_kw={'aspect': 'equal'})
    for i, (ax, d) in enumerate(zip(axs, [target, pred])):
        img = _create_plot(ax, d, k, max_value, axis, colmap)
        if i == 0:
            ax.set_title("Target")
        else:
            ax.set_title("Prediction")
    fig.colorbar(img, ax=axs)
    if show_title:
        fig.suptitle(fr"Reconstruction for fixed $k_{axis}$", fontsize=16)
