import numpy as np
import matplotlib.pyplot as plt

def midpoints(arr):
    return arr[:-1] + np.diff(arr)/2.0

def plot_hist_step(ax, edges, contents, errors, kwargs_step={}, kwargs_errorbar={}):
    line = histstep(ax, edges, contents, **kwargs_step)
    #ax.errorbar(midpoints(edges), contents, errors, lw=0, elinewidth=1, color=line.get_color()[0], **kwargs_errorbar)
    
def histstep(ax, edges, contents, **kwargs):
    ymins = []
    ymaxs = []
    xmins = []
    xmaxs = []
    for istep in range(len(edges)-1):
        xmins += [edges[istep]]
        xmaxs += [edges[istep+1]]
        ymins += [contents[istep]]
        if istep + 1 < len(contents):
            ymaxs += [contents[istep+1]]

    if not "color" in kwargs:
        kwargs["color"] = next(ax._get_lines.prop_cycler)['color']

    ymaxs += [ymaxs[-1]]
    l0 = ax.hlines(ymins, xmins, xmaxs, **kwargs)
    l1 = ax.vlines(xmaxs, ymins, ymaxs, color=l0.get_color(), linestyles=l0.get_linestyle())
    return l0

def plot_hist_ratio(hists_mc, hist_data,
        total_err_stat=None,
        total_err_stat_syst=None,
        figure=None, **kwargs):

    xbins = kwargs.get("xbins", None)

    if not figure:
        figure = plt.figure(figsize=(4,4), dpi=100)

    ax1 = plt.axes([0.0, 0.23, 1.0, 0.8])

    hmc_tot = np.zeros_like(hist_data.contents)
    hmc_tot2 = np.zeros_like(hist_data.contents)

    edges = hist_data.edges
    if xbins == "uniform":
        edges = np.arange(len(hist_data.edges))


    ax1.errorbar(
        midpoints(edges), hist_data.contents,
        np.sqrt(hist_data.contents_w2), marker="o", lw=0,
        elinewidth=1.0, color="black", ms=3, label="data")
    
    for h in hists_mc:
#         plot_hist_step(ax1, edges, hmc_tot + h.contents,
#             np.sqrt(hmc_tot2 + h.contents_w2),
#             kwargs_step={"label": getattr(h, "label", None), "color": getattr(h, "color", None)}
#         )

        b = ax1.bar(midpoints(edges), h.contents, np.diff(edges), hmc_tot,
            edgecolor=getattr(h, "color", None),
            facecolor=getattr(h, "color", None),
            label=getattr(h, "label", None))
        hmc_tot += h.contents
        hmc_tot2 += h.contents_w2

    ax1.errorbar(
        midpoints(edges), hist_data.contents,
        np.sqrt(hist_data.contents_w2), marker="o", lw=0,
        elinewidth=1.0, color="black", ms=3)

    if not (total_err_stat_syst is None):
        #plt.bar(midpoints(edges), 2*total_err_stat_syst, bottom=hmc_tot - total_err_stat_syst,
        #        width=np.diff(edges), hatch="////", color="gray", fill=None, lw=0)
        histstep(ax1, edges, hmc_tot + total_err_stat_syst, color="blue", linewidth=0.5, linestyle="--", label="stat+syst")
        histstep(ax1, edges, hmc_tot - total_err_stat_syst, color="blue", linewidth=0.5, linestyle="--")

    if not (total_err_stat is None):
        #plt.bar(midpoints(edges), 2*total_err_stat, bottom=hmc_tot - total_err_stat,
        #        width=np.diff(edges), hatch="\\\\", color="gray", fill=None, lw=0)
        histstep(ax1, edges, hmc_tot + total_err_stat, color="gray", linewidth=0.5, linestyle="--", label="stat")
        histstep(ax1, edges, hmc_tot - total_err_stat, color="gray", linewidth=0.5, linestyle="--")

    if kwargs.get("do_log", False):
        ax1.set_yscale("log")
        ax1.set_ylim(1, 100*np.max(hist_data.contents))
    else:
        ax1.set_ylim(0, 2*np.max(hist_data.contents))

    #ax1.get_yticklabels()[-1].remove()

    ax2 = plt.axes([0.0, 0.0, 1.0, 0.16], sharex=ax1)

    ratio = hist_data.contents / hmc_tot
    ratio_err = np.sqrt(hist_data.contents_w2) /hmc_tot
    ratio[np.isnan(ratio)] = 0

    plt.errorbar(
        midpoints(edges), ratio, ratio_err,
        marker="o", lw=0, elinewidth=1, ms=3, color="black")

    if not (total_err_stat_syst is None):
        ratio_up = (hmc_tot + total_err_stat_syst) / hmc_tot
        ratio_down = (hmc_tot - total_err_stat_syst) / hmc_tot
        ratio_down[np.isnan(ratio_down)] = 1
        ratio_down[np.isnan(ratio_up)] = 1
        histstep(ax2, edges, ratio_up, color="blue", linewidth=0.5, linestyle="--")
        histstep(ax2, edges, ratio_down, color="blue", linewidth=0.5, linestyle="--")

    if not (total_err_stat is None):
        ratio_up = (hmc_tot + total_err_stat) / hmc_tot
        ratio_down = (hmc_tot - total_err_stat) / hmc_tot
        ratio_down[np.isnan(ratio_down)] = 1
        ratio_down[np.isnan(ratio_up)] = 1
        histstep(ax2, edges, ratio_up, color="gray", linewidth=0.5, linestyle="--")
        histstep(ax2, edges, ratio_down, color="gray", linewidth=0.5, linestyle="--")


    plt.ylim(0.5, 1.5)
    plt.axhline(1.0, color="black")

    if xbins == "uniform":
        print(hist_data.edges)
        ax1.set_xticks(edges)
        ax1.set_xticklabels(["{0:.2f}".format(x) for x in hist_data.edges])

    ax1.set_xlim(min(edges), max(edges))
    xlim = kwargs.get("xlim", None)
    if not xlim is None:
        ax1.set_xlim(*xlim)

    ylim = kwargs.get("ylim", None)
    if not ylim is None:
        ax1.set_ylim(*ylim)

    return ax1, ax2
