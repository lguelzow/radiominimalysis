import iminuit
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from radiotools.atmosphere import models as atm
from scipy import stats
import matplotlib.ticker as tick 
import cmasher as cmr

from radiominimalysis.framework import factory
from radiominimalysis.framework.parameters import (
    eventParameters as evp,
    showerParameters as shp,
)
from radiominimalysis.utilities import (
    cherenkov_radius as che,
    helpers,
    stats as helperstats,
)

from lmfit import (Minimizer, Parameters, conf_interval, conf_interval2d,
                   report_ci, report_fit)


def evaluate_fitted_cherenkov_radius(events, para):

    plt.rcParams['font.size'] = 13

    mask = factory.has_parameter(events, shp.geomagnetic_ldf_parameter)
    print("Events with successful fit: %d / %d" % (np.sum(mask), len(mask)))
    events = events[mask]
    
    # cut events with small geomagnetic angles
    if 1:
        alpha = factory.get_parameter(events, shp.geomagnetic_angle)
        mask = alpha > 0.35
        print(f"Cut {np.sum(~mask)} events with geomagnetic angle < 20°")
        events = events[mask]

    if 0:
        energy_mc = factory.get_parameter(events, shp.energy)
        prim = factory.get_parameter(events, shp.primary_particle)
        print("Select only protons with lgE < 18.5")
        events = events[np.all([energy_mc < 10 ** 18.5, prim < 15], axis=0)]

    geo_ldf_params_fit = factory.get_parameter(events, shp.geomagnetic_ldf_parameter)
    if np.all(factory.has_parameter_error(events, shp.geomagnetic_ldf_parameter)):
        geo_ldf_params_fit_err = factory.get_parameter_error(
            events, shp.geomagnetic_ldf_parameter
        )
    else:
        geo_ldf_params_fit_err = None

    distance_to_xmax_mc = factory.get_parameter(
        events, shp.distance_to_shower_maximum_geometric
    )
    rho_mc = factory.get_parameter(events, shp.density_at_shower_maximum)
    zenith_mc = factory.get_parameter(events, shp.zenith)

    r0 = np.array([x["r0"] for x in geo_ldf_params_fit])
    if geo_ldf_params_fit_err is not None:
        r0_err = np.array([x["r0"] for x in geo_ldf_params_fit_err])

    cherenkov_angle_rec = np.arctan(r0 / distance_to_xmax_mc)

    atmodels = factory.get_parameter(events, shp.atmosphere_model)
    n0s = factory.get_parameter(events, evp.refractive_index_at_sea_level)
    obs_levels = factory.get_parameter(events, shp.observation_level)

    use_rho = False
    fig, axs = plt.subplots(2, sharex=True)
    idx = 0
    for model in np.unique(atmodels)[::-1]:
        at = atm.Atmosphere(model)
        for n0 in np.unique(n0s):
            for obs in np.unique(obs_levels):

                mask = np.all([obs_levels == obs, n0s == n0, atmodels == model], axis=0)

                if not np.any(mask):
                    continue
                print("ATM Model: ", model, ", Air Density at sea level:", n0, ", Number of Events: ", np.sum(mask))

                h_asl = (
                    atm.get_height_above_ground(
                        d=distance_to_xmax_mc[mask],
                        zenith=zenith_mc[mask],
                        observation_level=obs,
                    )
                    + obs
                )

                delta = che.get_cherenkov_angle_model(h_asl, n0, model)
                n_h = atm.get_n(h_asl, n0=n0, model=model)
                delta_test = np.arccos(1 / n_h)
                alpha = np.amin([0.2, 200 / np.sum(mask)])

                if use_rho:
                    xval = rho_mc[mask]
                else:
                    xval = distance_to_xmax_mc[mask]

                yval = (cherenkov_angle_rec[mask] / delta)

                # axs[1].plot(xval, (cherenkov_angle_rec[mask] / delta), "o", alpha=alpha)

                # axs[1].set_ylim(0.87, 0.94)
                # axs[1].set_xlim(89000, 112000)

                # get quantity for colormap
                energy = factory.get_parameter(events, shp.energy)
                # colormap = energy
                # print(colormap)
                

                # parameter = axs[1].scatter(xval / 1000, cherenkov_angle_rec[mask] / delta, marker="o", s=10, alpha=0.6, c="orange")
                parameter = axs[1].scatter(xval / 1000, cherenkov_angle_rec[mask] / delta, marker="o", s=10, alpha=0.2, cmap="plasma", c=np.log10(energy[mask]), vmin=18.4, vmax=20.2)

                 # add colourbar for energy colormap
                divider = make_axes_locatable(axs[1])
                cax = divider.append_axes("right", size="3%", pad=0.05)
                # assign colorbar to this subplot
                cbar = plt.colorbar(parameter, ax=axs[1], cax=cax, format=tick.FormatStrFormatter('%.1f'))
                cbar.set_label(r"log(Energy / EeV)")

                # print("Mean of colormap: ", np.mean(colormap))

                # define parametrisation function
                def r_0_deviation_model(dmax, c0, c1, c2):
                    return c0 + c1 * dmax + c2 / (dmax ** 2)

                # least square fit; fit function
                def residual(params, dmax, data):
                    c0 = params['c0']
                    c1 = params['c1']
                    c2 = params['c2']

                    return (data - r_0_deviation_model(dmax, c0, c1, c2)) / np.sqrt(data)

                # calculate the chisq for the model p = 1/d
                params = Parameters()
                params.add('c0', value=0.91, vary=True)
                params.add('c1', value=2.85542730e-06, vary=True)
                params.add('c2', value=1.87269374e+07, vary=True)

                mini = Minimizer(residual, params, fcn_args=(xval, yval))
                result0 = mini.leastsq()

                fit_res0 = residual(result0.params, xval, yval)
                report_fit(result0)

                ci = conf_interval(mini, result0)
                # print(ci)
                report_ci(ci)

                dmax_range = np.linspace(5000, max(xval) * 1.2, 500)
                r0_dev_param = r_0_deviation_model(dmax=dmax_range, c0=result0.params['c0'].value, c1=result0.params['c1'].value, c2=result0.params['c2'].value)
                # plot the fit
                axs[1].plot(dmax_range / 1000, r0_dev_param, color='black', ls="--", label=r'Parametrisation of $\delta_0$ deviation', linewidth=1)  # , alpha=0.5)
                
                # print(xval)
                # print(fit_same)
                
                # indices determining confidence levels
                index_high = 6
                index_low = 0

                # define confidence interval functions
                # get coincidence interval values from ci:
                # 1st index determines confidence level (0: 99.73%; 1: 95.45%; 2: 68.27%, 3: BEST VALUE)
                # 1 - 3 are the low bounds and 4 - 6 are the high bounds (reverse order)
                # 2nd index gives you the value of the confidence level

                ci_r0_dev_high = r_0_deviation_model(dmax=dmax_range, c0=ci['c0'][index_high][1], c1=ci['c1'][index_high][1], c2=ci['c2'][index_high][1])
                
                ci_r0_dev_low = r_0_deviation_model(dmax=dmax_range, c0=ci['c0'][index_low][1], c1=ci['c1'][index_low][1], c2=ci['c2'][index_low][1])

                # plot confidence regions
                # plt.fill_between(dmax_range / 1000, ci_r0_dev_low, ci_r0_dev_high, interpolate=True, color='red', alpha=0.3, label=r"$5\sigma$ Confindence Interval")


                axs[0].plot(
                    xval / 1000,
                    np.rad2deg(cherenkov_angle_rec)[mask],
                    "o",
                    alpha=alpha,
                    color="red", ms=4 # orangered
                )
                axs[0].plot(
                    np.nan,
                    np.nan,
                    "o",
                    color="red", # orangered
                    label=r"$\delta_0^\mathrm{fit}$"
                    # label=r"ATM: %d, n$_0$ = %.2e" % (model, n0 - 1),
                )

                # plot cherenkov angle for matching rounded zeniths
                # 5 and 6 for GRAND
                # 4 and 4 for Auger
                for zen in np.unique(np.around(zenith_mc[mask], 5)): # 4
                    mask2 = np.around(zenith_mc[mask], 6) == zen # 4

                    if not np.any(mask):
                        continue

                    dmax_mean = np.mean(xval[mask2])
                    dmax_std = np.std(xval[mask2])
                    delta_mean = np.mean((cherenkov_angle_rec[mask] / delta)[mask2])
                    delta_std = np.std((cherenkov_angle_rec[mask] / delta)[mask2])

                    axs[0].plot(
                        xval[mask2] / 1000, np.rad2deg(delta)[mask2], "-", color="black"
                    )
                    # axs[1].errorbar(
                    #     dmax_mean,
                    #     delta_mean,
                    #     delta_std,
                    #     dmax_std,
                    #     marker="o",
                    #     ls="",
                    #     color="C%d" % idx,
                    #     capsize=5,
                    #     zorder=10,
                    # )
                idx += 1

    if use_rho:
        axs[1].set_xlabel(r"$\rho_\mathrm{max}^\mathrm{MC}$ [kg$\,$m$^{-3}]$")
    else:
        axs[1].set_xlabel(r"$d_\mathrm{max}^\mathrm{MC}$ [km]")

    axs[0].plot(np.nan, np.nan, label=r"$\delta^\mathrm{pred}_0(n(h_\mathrm{max}))$", c="black")
    axs[0].set_ylabel(r"$\delta_0$ [°]")
    axs[1].set_ylabel(
        r"$\delta_0^\mathrm{fit} / \delta^\mathrm{pred}_0$"
    )
    axs[0].legend(ncol=3, fontsize=13) #, fontsize=20)
    axs[1].legend(ncol=3, fontsize=13)
    axs[0].set_xlim(0, 230) # 180
    axs[0].set_ylim(0.24, 1.03)
    axs[1].set_ylim(0.59, 1.03)
    [ax.grid() for ax in axs]
    [ax.tick_params(labelsize=13) for ax in axs]
    axs[0].set_title("GP300", fontsize=16)
    plt.tight_layout()
    plt.savefig("Cherenkov_comparison.png", dpi=300)
    plt.show()
    
    
def r0_param(r0_start, x, a0, a1, a2,):
    dmax = x
    return r0_start * (a0 + (a1 * dmax) + (a2 / dmax / dmax))


def r02_arel_param(x, a0, a1, a2, a3=0):
    dmax, dmax_avg = x
    return a0 + a1 * dmax + a2 / dmax ** 2 + a3 * (dmax - dmax_avg)
    # return a0 + a1 * dmax + a2 / dmax ** 2 + a3 * (np.log10(dmax) - np.log10(dmax_avg))


def r02_arel_param2(x, a0, a1, a2):
    dmax = x
    return a0 + a1 * dmax + a2 / dmax


def r02_arel_param_rho(x, a0, a1, a2, a3=0):
    rho, rho_avg = x
    return a0 + a1 * rho + a2 * rho ** 2 + a3 * (rho - rho_avg)


def sig_param(x, a, b, c, d=0):
    dmax, dmax_avg = x
    return a * (dmax - 5e3) ** b + c + d * (dmax - dmax_avg)


def p_param(x, a, b, c, d=0):
    dmax, dmax_avg = x
    return a * np.exp(-b * dmax) + c + d * (dmax - dmax_avg)
    # return a / (1 + np.exp(-b * (x - c)))


def p_param_new_rho(rho, a, b):
    return a * rho + b


def r02_param_rho_new(rho, a, b, c=1):
    return a * np.exp(-b * rho) + c


def r02_power_law(x, a, b, c, d=0):
    dmax, dmax_avg = x
    return a * (dmax - 5e3) ** b + c + d * (dmax - dmax_avg)

def r02_exponential(x, a, b, c): 
    dmax = x
    return a - b * np.exp(-dmax * c)


def objective_func_leastsq(pars, func, x, y, yerr):
    ypred = func(x, *pars)
    return np.sum(np.square((y - ypred) / yerr))


def get_param_and_uncert(res):
    mini = res.minuit
    popt = np.array(mini.values)
    uncerts = np.array(mini.errors)
    return popt, uncerts


def get_bins_for_x_from_binned_data(xval, binned_data):
    bins = [xval.min()]
    for y in np.unique(binned_data):
        mask = y == binned_data
        min_tmp = xval[mask].min()
        if bins[-1] < min_tmp:
            bins.append(min_tmp)
        bins.append(xval[mask].max())
    return np.array(bins)


def _plot_profile(ax, x, y, xbins, color=None):
    n, xcen, y_mean_binned, y_std_binned, edges = helperstats.get_binned_data(
        x, y, xbins, skip_empty_bins=False, return_bins=True
    )

    xerr = np.array([xcen - edges[:-1], edges[1:] - xcen])
    mask = n > 5
    ax.errorbar(
        xcen[mask],
        y_mean_binned[mask],
        y_std_binned[mask],
        xerr[:, mask],
        markersize=8,
        color="black",
        markerfacecolor="red",
        markeredgewidth=1.5,
        marker="s",
        ls="",
        zorder=3,
        alpha=1
    )


def plot_profile(ax, x, y, xbins=10, colors=None):

    if colors is not None and len(np.unique(colors)) != 1:
        for c in np.unique(colors):
            scolor = "C%d" % c
            print("scolor", scolor)
            cmask = colors == c
            _plot_profile(ax, x[cmask], y[cmask], xbins=xbins, color="maroon")

    else:
        _plot_profile(ax, x, y, xbins=xbins, color="limegreen")


def _plot(ax, xplot, y, yerr, colors=None, res=False, colormap=None, plot_colormap=False):
    
    color_points = "limegreen"
    colourscheme = "viridis"

    # print("is it empty?", colormap)

    if colors is not None and len(np.unique(colors)) != 1:
        for c in np.unique(colors):
            scolor = "C%d" % c
            print("scolor", scolor)
            cmask = colors == c
            alpha = max((len(colors) - np.sum(cmask)) / len(colors) / 3, 0.1)

            if not np.all(np.ones_like(yerr) == yerr):
                ax.errorbar(
                    xplot[cmask],
                    y[cmask],
                    yerr[cmask],
                    color=color_points,
                    marker="o",
                    ls="",
                    alpha=alpha,
                )
            else:
                if res:
                    cor = stats.pearsonr(xplot[cmask], y[cmask])
                    ax.plot(
                        xplot[cmask],
                        y[cmask],
                        "o",
                        color=color_points,
                        alpha=alpha,
                        label=r"$\mu$ = %.3f, $\sigma$ = %.3f, corr =  %.3f"
                        % (np.mean(y[cmask]), np.std(y[cmask]), cor[0]),
                    )
                else:
                    ax.plot(xplot[cmask], y[cmask], "o", color=color_points, alpha=alpha)
    else:
        if not np.all(np.ones_like(yerr) == yerr):
            
            if plot_colormap == False:
                ax.errorbar(xplot, y, yerr, c=color_points, marker="o", ls="", alpha=0.5)

            else:

                # add colourbar for xmax colormap
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)

                parameter = ax.scatter(xplot, y, marker="o", s=25, alpha=0.4, cmap=colourscheme, c=colormap, vmin=18.4, vmax=20)

                # assign colorbar to this subplot
                cbar = plt.colorbar(parameter, ax=ax, cax=cax, format=tick.FormatStrFormatter('%.1f'))
                cbar.set_label(r"log(Energy / EeV)")

                # print("Mean of colormap: ", np.mean(colormap))


        else:
            if res:
                cor = stats.pearsonr(xplot, y)
                ax.plot(
                    xplot,
                    y,
                    "C0o",
                    alpha=0.1,
                    label=r"$\mu$ = %.3f, $\sigma$ = %.3f, corr =  %.3f"
                    % (np.mean(y), np.std(y), cor[0]),
                )
            else:
                ax.plot(xplot, y, "o", alpha=0.1)


def plot_parameter(
    ax,
    x,
    y,
    yerr,
    ylabel,
    colors=None,
    func=None,
    p0=None,
    xbins=None,
    pl_profile=False,
    plot_param=False,
    xlabel=None,
    ax_res=None,
    x_avg=None,
    ax_res_xlabel=None,
    plot_avg=False,
    ylim=None,
    colormap=None,
    plot_colormap=False
):

    # use some quantity as colormap if given 
    quantity_map = colormap
    # print(quantity_map)

    x = np.copy(x)
    y = np.copy(y)
    yerr = np.copy(yerr)

    xmodel = helpers.get_fine_xs(x)
    if plot_param:
        try:
            res = iminuit.minimize(
                objective_func_leastsq, x0=p0, args=(func, x, y, yerr)
            )
            popt, uncert = get_param_and_uncert(res)

            val_popt = func(xmodel, *popt)
            val_p0 = func(xmodel, *p0)
        except ValueError:
            res = iminuit.minimize(
                objective_func_leastsq, x0=p0, args=(func, [x, x_avg], y, yerr)
            )
            popt, uncert = get_param_and_uncert(res)

            val_popt = func([x, x_avg], *popt)
            val_p0 = func([x, x_avg], *p0)

        print(ylabel, popt, uncert / popt)
        # print(np.any(np.isnan(uncert)))

    else:
        popt = None

    if xmodel.max() > 10e3:
        xmodel /= 1e3
        xplot = x / 1e3
    else:
        xplot = x

    _plot(ax, xplot, y, yerr, colors, colormap=quantity_map, plot_colormap=plot_colormap)

    ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if pl_profile:
        plot_profile(ax, xplot, y, xbins, colors)

    if plot_param and not np.any(np.isnan(uncert)):
        try:
            # ax.plot(xmodel, val_p0, "-", color="lightgrey", alpha=0.5, zorder=10)
            ax.plot(xmodel, val_popt, "k-", zorder=10)
        except ValueError:
            # ax.plot(xplot, val_p0, ".", color="lightgrey", alpha=0.5, zorder=10)
            ax.plot(xplot, val_popt, "k.", zorder=10)

    if ax_res is not None and plot_param and not np.any(np.isnan(uncert)):
        try:
            res = (y - func(x, *popt)) / func(x, *popt)
        except ValueError:
            res = (y - func([x, x_avg], *popt)) / func([x, x_avg], *popt)

        if x_avg is not None and plot_avg:
            if x_avg.max() > 10e3:
                xplot -= x_avg / 1e3
            else:
                xplot -= x_avg

        _plot(ax_res, xplot, res, np.ones_like(res), colors, res=True, plot_colormap=False)

        # If not "plot_avg" overwrite "ax_res_xlabel" anyway
        if ax_res_xlabel is None or not plot_avg:
            ax_res_xlabel = xlabel

        if ax_res_xlabel is not None:
            ax_res.set_xlabel(ax_res_xlabel)

    if ylim is not None:
        ax.set_ylim(ylim)

    return popt


def get_r0_param_and_avg_data_and_colors(events, cal_avg=True):

    avg_xmax = 750

    atmodels = factory.get_parameter(events, shp.atmosphere_model)
    n0s = factory.get_parameter(events, evp.refractive_index_at_sea_level)
    obs_levels = factory.get_parameter(events, shp.observation_level)
    zenith_mc = factory.get_parameter(events, shp.zenith)
    dmax_mc = factory.get_parameter(events, shp.distance_to_shower_maximum_geometric)

    r0_param = np.zeros_like(zenith_mc)
    rho_mc_avg = np.zeros_like(zenith_mc)
    dmax_mc_avg = np.zeros_like(zenith_mc)
    colors = np.zeros_like(zenith_mc)

    cdx = 0
    for model in np.unique(atmodels)[::-1]:
        at = atm.Atmosphere(model)
        for n0 in np.unique(n0s):
            for obs in np.unique(obs_levels):

                mask = np.all([obs_levels == obs, n0s == n0, atmodels == model], axis=0)

                if not np.any(mask):
                    continue

                colors[mask] = np.ones(np.sum(mask)) * cdx

                print(model, n0, obs, cdx)
                cdx += 1
                r0_param[mask] = che.get_cherenkov_radius_model_from_distance(
                    zenith_mc[mask], dmax_mc[mask], obs, n0, model
                )

                if not cal_avg:
                    continue

                for zen in np.unique(np.around(zenith_mc, 4)):
                    mask2 = np.all([np.around(zenith_mc, 4) == zen, mask], axis=0)
                    h = at.get_vertical_height(zen, avg_xmax, obs)
                    d = atm.get_distance_for_height_above_ground(h - obs, zen, obs)
                    rho_tmp = atm.get_density(h, model=model) * 1e-3
                    rho_mc_avg[mask2] = np.ones(np.sum(mask2)) * rho_tmp
                    dmax_mc_avg[mask2] = np.ones(np.sum(mask2)) * d

    print("Colours", colors)

    return r0_param, dmax_mc_avg, rho_mc_avg, colors


def draw_offline_param(axs, dmax, fmt):
    # param which was used for first version of gaus+sigmoid

    # sig
    axs[1, 0].plot(
        dmax / 1e3,
        sig_param((dmax, dmax), *[0.16848311, 0.69447957, 39.81137662]),
        fmt,
        zorder=10,
    )
    # r02
    axs[0, 1].plot(
        dmax / 1e3,
        r02_arel_param((dmax, dmax), *[5.33720788e-01, 8.37285946e-07, 5.27475125e07]),
        fmt,
        zorder=10,
    )

    # p
    p = lambda x: 1.85054143e00 / (1 + np.exp(-4.20849856e-05 * (x + 2.86110554e04)))
    axs[1, 1].plot(dmax / 1e3, p(dmax), fmt, zorder=10)

    # arel
    axs[0, 2].plot(
        dmax / 1e3,
        r02_arel_param((dmax, dmax), *[7.47991658e-01, 8.11304687e-07, 1.72095209e07]),
        fmt,
        zorder=10,
    )


def draw_hard_param(axs, dmax, fmt, label='"hard p"'):
    # new param with hard p and without d750

    # sig
    axs[1, 0].plot(
        dmax / 1e3,
        sig_param((dmax, dmax), *[0.16460107, 0.69675388, 39.46602294]),
        fmt,
        zorder=10,
    )
    # wo d750 0.16460107  0.69675388 39.46602294
    # with d750: 0.174353946  0.692199262. 37.0561452

    # p
    axs[1, 1].plot(
        dmax / 1e3,
        p_param((dmax, dmax), *[-4.94492500e-01, 3.90450109e-05, 1.85763162e00]),
        fmt,
        zorder=10,
    )
    # wo d750: -4.94492500e-01  3.90450109e-05  1.85763162e+00
    # with d750: -4.66078147e-01, 3.60604450e-05, 1.86010517e+00

    # arel
    axs[0, 2].plot(
        dmax / 1e3,
        r02_arel_param((dmax, dmax), *[7.57300343e-01, 7.76904953e-07, 1.63545932e07]),
        fmt,
        zorder=10,
    )
    # wo d750: 7.57300343e-01 7.76904953e-07 1.63545932e+07
    # with d750: 7.56127634e-01 7.67832252e-07 1.70017091e+07 (2.86813527e-06)
    # with d750 (only sig): 7.56920056e-01 7.69145360e-07 1.61848905e+07

    # r02
    axs[0, 1].plot(
        dmax / 1e3,
        r02_arel_param((dmax, dmax), *[5.48029929e-01, 7.06584317e-07, 5.06592690e07]),
        fmt,
        zorder=10,
    )
    # wo d750: 5.48029929e-01 7.06584317e-07 5.06592690e+07
    # with d750: 5.40745182e-01 7.08091188e-07 5.00073845e+07
    # with d750 (only sig): 5.49448820e-01 6.98439471e-07 4.95005832e+07

    axs[1, 2].plot(np.nan, np.nan, fmt, label=label)


def draw_soft_param(axs, r0_start, dmax, dmax_avg, fmt, label='"soft p"', plot_p=True, linewidth=3):
    # new param with soft p. param is with d750 (sig arel), here drawn without it
    
    # r0
    # axs[0, 0].plot(
    #     dmax / 1e3,
    #     r0_param(r0_start, dmax, *[0.81810574,5.6302e-07,-36815284.6]),
    #     fmt,
    #     zorder=10, ls="--", lw=linewidth
    # )
    # Auger 0.94061131,-2.2048e-07,-15960366.4
    # GRAND 0.81810574,5.6302e-07,-36815284.6
    
    # sig
    axs[1, 0].plot(
        dmax / 1e3,
        sig_param((dmax, dmax_avg), *[2.71147327e-02,8.05075351e-01,6.19714390e+01]),
        fmt,
        zorder=10, ls="--", lw=linewidth
    )
    # Auger 2.71147327e-02,8.05075351e-01,6.19714390e+01
    # GRAND 3.94026614e-02,7.59980861e-01,6.09984088e+01

    # r02
    axs[0, 1].plot(
        dmax / 1e3,
        r02_arel_param((dmax, dmax_avg), *[5.98281908e-01,1.29545941e-06,-9.86454075e+07]),
        fmt,
        zorder=10, ls="--", lw=linewidth
    )
    # Auger 5.98281908e-01,1.29545941e-06,-9.86454075e+07
    # GRAND 5.60308106e-01,1.04413624e-06,-1.43854800e+08

    # b
    if plot_p:
        axs[1, 1].plot(
            dmax / 1e3,
            r02_arel_param((dmax, dmax_avg), *[2.82212329e+02,-3.65695435e-04,-6.45711498e+09]), 
            fmt,
            zorder=10, ls="--", lw=linewidth
        )
        # Auger 2.82212329e+02,-3.65695435e-04,-6.45711498e+09
        # GRAND 2.46529404e+02,-2.23065116e-04,-9.29903285e+09

    # arel
    axs[0, 2].plot(
        dmax / 1e3,
        r02_arel_param((dmax, dmax_avg), *[2.32868923e-01,2.06291784e-07,-3.78691182e+06]),
        fmt,
        zorder=10, ls="--", lw=linewidth
    )
    # Auger 2.32868923e-01,2.06291784e-07,-3.78691182e+06
    # GRAND 2.65672136e-01,5.74786264e-07,-1.92805026e+07

    axs[1, 2].plot(np.nan, np.nan, fmt, label=label)

    # p_slope
    axs[0, 3].plot(
        dmax / 1e3,
        r02_arel_param((dmax, dmax_avg), *[1.46438531e+00,-5.26817171e-08,3.29440576e+07]), 
        fmt,
        zorder=10, ls="--", lw=linewidth
    )

    # Auger 1.46438531e+00,-5.26817171e-08,3.29440576e+07
    # GRAND 1.52037333e+00,3.82623066e-07,-4.26437099e+07


def evaluate_gauss_sigmoid_pars(events, para):
    # is_offline = ~np.all(has_shower)
    # has_shower = np.array([ev.has_shower(evp.rd_shower) for ev in events])
    # print("Events with Rd shower: %d / %d" %
    #       (np.sum(has_shower), len(has_shower)))
    # events = events[has_shower]

    # globally change fontsize for plots
    plt.rcParams['font.size'] = 32

    mask = factory.has_parameter(events, shp.geomagnetic_ldf_parameter)
    # print(mask)
    print("Events with successful fit: %d / %d" % (np.sum(mask), len(mask)))
    events = events[mask]
    
    # cut events with small geomagnetic angles
    if 1:
        alpha = factory.get_parameter(events, shp.geomagnetic_angle)
        mask = alpha > 0.35
        print(f"Cut {np.sum(~mask)} events with geomagnetic angle < 20°")
        events = events[mask]

    if "valid" in factory.get_parameter(events, shp.fit_result)[0]:
        valid = np.array(
            [x["valid"] for x in factory.get_parameter(events, shp.fit_result)]
        )
        print("Events with valid fit: %d / %d" % (np.sum(valid), len(valid)))
        events = events[valid]

    if np.all(factory.has_parameter_error(events, shp.geomagnetic_ldf_parameter)):
        dmax_fit, dmax_fit_err = factory.get_parameter_and_error(
            events, shp.distance_to_shower_maximum_geometric_fit
        )

        mask_none = dmax_fit_err != None
        print(
            "Events with rel dmax err != None: %d / %d"
            % (np.sum(mask_none), len(mask_none))
        )
        mask = dmax_fit_err[mask_none] / dmax_fit[mask_none] < 1
        print("Events with rel dmax err < 1: %d / %d" % (np.sum(mask), len(mask)))
        events = events[mask_none][mask]
        geo_ldf_params_fit_err = factory.get_parameter_error(
            events, shp.geomagnetic_ldf_parameter
        )

        # every parameter should work here
        egeo_err = np.array([x["E_geo"] for x in geo_ldf_params_fit_err])
        mask = ~np.isnan(egeo_err)
        print("Events with nan error: %d / %d" % (np.sum(mask), len(mask)))
        events = events[mask]

    if 0:
        geo_ldf_params_fit = factory.get_parameter(
            events, shp.geomagnetic_ldf_parameter
        )
        r02 = np.array([x["r02"] for x in geo_ldf_params_fit])
        mask = r02 < 1 - 1e-2

        print(np.array([e.get_run_number() for e in events])[~mask])
        print("Events with r02 < 1: %d / %d" % (np.sum(mask), len(mask)))
        events = events[mask]

    if 0:
        fit_result = factory.get_parameter(events, shp.fit_result)
        dmax_mc = factory.get_parameter(
            events, shp.distance_to_shower_maximum_geometric
        )

        redchi = np.array([ele["redchi"] for ele in fit_result])
        mask = ~np.all([redchi > 2, dmax_mc > 120e3], axis=0)
        print("Events dmax > 120e3 and redchi > 2: %d / %d" % (np.sum(mask), len(mask)))
        events = events[mask]

    xmax_mc = factory.get_parameter(events, shp.xmax)
    # print(np.mean(xmax_mc))

    filter_for_avg_xmax = False
    # make and apply mask for only looking at showers with xmax near the avg of 750 g/cm²
    if filter_for_avg_xmax == True:
        mask = np.all([xmax_mc / 750 > 0.95, xmax_mc / 750 < 1.05], axis=0)
        events = events[mask]
        xmax_mc = xmax_mc[mask]

    dmax_mc = factory.get_parameter(events, shp.distance_to_shower_maximum_geometric)
    rho_mc = factory.get_parameter(events, shp.density_at_shower_maximum)
    zenith_mc = factory.get_parameter(events, shp.zenith)
    energy = factory.get_parameter(events, shp.energy)
    r0_start = factory.get_parameter(events, shp.r0_start)
    
    r0_test = r0_start * (0.81810574 + (5.6302e-07 * dmax_mc) + ((-36815284.6)) / dmax_mc / dmax_mc)
    

    para_rho = False
    plot_param = True
    pl_profile = True
    show_colormap = True
    res_avg = True
    use_fitted_errors = True
    new_p = True
    plot_single = False # True: plot each parameter in its own plot, False: grouped plot
    ssize = (45, 20)

    colormap = np.log10(energy) # np.rad2deg(geomagnetic_angle)

    fit_params = factory.get_parameter(events, shp.geomagnetic_ldf_parameter)
    if (
        np.all(factory.has_parameter_error(events, shp.geomagnetic_ldf_parameter))
        and use_fitted_errors
    ):
        fit_params_err = factory.get_parameter_error(
            events, shp.geomagnetic_ldf_parameter
        )
    else:
        fit_params_err = np.array([{key: 1 for key in fit_params[0]}] * len(dmax_mc))

    arel = np.array([x["arel"] for x in fit_params])
    slope = np.array([x["slope"] for x in fit_params])
    r0 = np.array([x["r0"] for x in fit_params])
    r02 = np.array([x["r02"] for x in fit_params])
    p = np.array([x["p"] for x in fit_params])
    sig = np.array([x["sig"] for x in fit_params])
    p_slope = np.array([x["p_slope"] for x in fit_params])

    print("Averages of parameters:")
    print("a_rel: ", np.average(arel))
    print("r02: ", np.average(r02))
    print("p: ", np.average(p))
    print("p_slope: ", np.average(p_slope))
    print("sig: ", np.average(sig))
    print("sig_slope: ", np.average(slope))

    arel_err = np.array([x["arel"] for x in fit_params_err])
    slope_err = np.array([x["slope"] for x in fit_params_err])
    r0_err = np.array([x["r0"] for x in fit_params_err])
    r02_err = np.array([x["r02"] for x in fit_params_err])
    p_err = np.array([x["p"] for x in fit_params_err])
    sig_err = np.array([x["sig"] for x in fit_params_err])
    p_slope_err = np.array([x["p_slope"] for x in fit_params_err])

    dmax_lab = r"$d_\mathrm{max}^\mathrm{MC}$ [km]"
    dmax_lab_avg = r"$d_\mathrm{max}^\mathrm{MC} - d_\mathrm{750}^\mathrm{MC}$ / km"
    rho_lab = r"$\rho_\mathrm{max}^\mathrm{MC}$ / g$\,$cm$^{-2}$"
    rho_lab_avg = r"$\rho_\mathrm{max}^\mathrm{MC} - \rho_\mathrm{750}^\mathrm{MC}$ / g$\,$cm$^{-2}$"

    dmax_bins = np.sort(get_bins_for_x_from_binned_data(
        dmax_mc / 1e3, np.around(np.rad2deg(zenith_mc), 0)))

    rho_bins = np.sort(
        get_bins_for_x_from_binned_data(rho_mc, np.around(np.rad2deg(zenith_mc), 1))
    )

    r0_param, dmax_mc_avg, rho_mc_avg, colors = get_r0_param_and_avg_data_and_colors(
        events, True
    )

    # print("average dmax", dmax_mc_avg)

    if plot_single:
        fig, axs = plt.subplots(2, 4, figsize=ssize)
        fig_res, axs_res = plt.subplots(2, 4, figsize=ssize)

    else:
        fig, axs = plt.subplots(2, 4, figsize=ssize)
        fig_res, axs_res = plt.subplots(2, 4, figsize=ssize)


    plot_parameter(
        axs[0, 0],
        dmax_mc,
        r0,
        r0_err, # wrong errror, just to display color map
        r"Gaussian peak position $r_0$ [m]",
        xlabel=dmax_lab,
        colors=colors,
        colormap=colormap,
        plot_colormap=show_colormap,
    )

    # if plot_param:
    #     axs[0, 0].plot(dmax_mc / 1e3, r0_param, "k.")
    #     axs_res[0, 0].plot(dmax_mc / 1e3, (r0_param - r0) /
    #                        r0_param, "C0o", alpha=0.1)

    if not 0:
        sig_popt = plot_parameter(
            axs[1, 0],
            dmax_mc,
            sig,
            sig_err,
            colors=colors,
            func=sig_param,
            pl_profile=pl_profile,
            xbins=dmax_bins,
            p0=[2.82209732e-02,8.00859075e-01,6.12971298e+01], # Auger Param
            # p0 = [3.50114248e-02,7.69590775e-01,6.29361743e+01],  # GRAND param
            plot_param=False, # plot_param,
            ylabel=r"Gaussian width $\sigma$ [m]",
            xlabel=dmax_lab,
            ax_res_xlabel=dmax_lab_avg,
            x_avg=dmax_mc_avg,
            plot_avg=res_avg,
            ax_res=axs_res[1, 0],
            # ylim=[50, 600],
            colormap=colormap,
            plot_colormap=show_colormap
        )

    else:
        sig_popt = plot_parameter(
            axs[1, 0],
            rho_mc,
            sig,
            sig_err,
            colors=colors,
            #  func=sig_param, pl_profile=pl_profile, xbins=dmax_bins,
            #  p0=[0.11514358, 0.72509144, 50.91115447],
            plot_param=False,
            ylabel=r"$\sigma$ / m",
            xlabel=rho_lab,
            ax_res_xlabel=rho_lab_avg,
            x_avg=rho_mc_avg,
            plot_avg=res_avg,
            ax_res=axs_res[1, 0],
            ylim=[50, 510],
        )

    if not para_rho:
        r02_popt = plot_parameter(
            axs[0, 1],
            dmax_mc,
            r02,
            r02_err,
            colors=colors,
            func=r02_arel_param,
            pl_profile=pl_profile,
            xbins=dmax_bins,
            # p0=[6.66316714e-01,2.62444830e-07,-6.49935124e+07], # Auger param
            p0=[5.85629629e-01,8.50072265e-07,-1.66284834e+08], # GRAND param
            # p0=[100, 0.1, -0.5], # for exp form
            ylabel=r"Sigmoid length scale $r_{02}$ [$r_0$]",
            plot_param=plot_param,
            xlabel=dmax_lab,
            ax_res_xlabel=dmax_lab_avg,
            x_avg=dmax_mc_avg,
            plot_avg=res_avg,
            ax_res=axs_res[0, 1],
            ylim=[0, 1],
            colormap=colormap,
            plot_colormap=show_colormap
        )

    else:
        r02_popt = plot_parameter(
            axs[0, 1],
            rho_mc,
            r02,
            r02_err,
            colors=colors,
            pl_profile=pl_profile,
            xbins=rho_bins,
            func=r02_arel_param_rho,
            p0=[5.45164084e-01, 1.18724857e-06, -1.34437089e+08],
            #  p0=[0.87789526, -2.10422792, 4.00626784],
            #  func=r02_param_rho_new,
            #  p0=[2834.97515674, 10.21176029, 202.19502505],
            plot_param=plot_param,
            # ylabel=r"$r_{02}$ [m]",
            ylabel=r"$r_{02}$ [$r_0$]",
            xlabel=rho_lab,
            ax_res_xlabel=rho_lab_avg,
            x_avg=rho_mc_avg,
            plot_avg=res_avg,
            ax_res=axs_res[0, 1.05],
            ylim=[0, 1.5],
        )

    if not 0:
        p_popt = plot_parameter(
            axs[1, 1],
            dmax_mc,
            p,
            p_err,
            colors=colors,
            func=r02_arel_param,
            pl_profile=pl_profile,
            xbins=dmax_bins,
            p0=[2.90121335e+02,-3.40459094e-04,-7.95160431e+09],  # Auger param
            # p0=[2.47077170e+02, -2.06317205e-04, -8.93411554e+09],  # GRAND param
            plot_param=False, # plot_param,
            ylabel=r"Outer Gaussian slope $b$",
            xlabel=dmax_lab,
            ax_res_xlabel=dmax_lab_avg,
            x_avg=dmax_mc_avg,
            plot_avg=res_avg,
            ax_res=axs_res[1, 1],
            ylim=[175, 350],
            colormap=colormap,
            plot_colormap=show_colormap
        )
    else:
        p_popt = plot_parameter(
            axs[1, 1],
            rho_mc,
            p,
            p_err,
            colors=colors,
            func=p_param_new_rho,
            pl_profile=pl_profile,
            xbins=rho_bins,
            p0=[1.88980566e+02, 6.09216147e-05, 2.02829398e+02],
            plot_param=plot_param,
            ylabel=r"$b$",
            xlabel=rho_lab,
            ax_res_xlabel=rho_lab_avg,
            x_avg=rho_mc_avg,
            plot_avg=res_avg,
            ax_res=axs_res[1, 1],
            ylim=[0, 500],
        )

    if not para_rho:
        arel_popt = plot_parameter(
            axs[0, 2],
            dmax_mc,
            arel,
            arel_err,
            colors=colors,
            func=r02_arel_param,
            pl_profile=pl_profile,
            xbins=dmax_bins,
            p0=[2.26817787e-01,3.07137877e-07,5.57521446e+04], # Auger param
            # p0=[2.64147015e-01, 5.08310712e-07, -1.30682037e+07], # GRAND param
            plot_param=False, #plot_param,
            ylabel=r"Sigmoid rel. amplitude $a_\mathrm{rel}$",
            xlabel=dmax_lab,
            ax_res_xlabel=dmax_lab_avg,
            x_avg=dmax_mc_avg,
            plot_avg=res_avg,
            ax_res=axs_res[0, 2],
            ylim=[0, 0.5],
            colormap=colormap,
            plot_colormap=show_colormap
        )

    else:
        arel_rho_popt = plot_parameter(
            axs[0, 2],
            rho_mc,
            arel,
            arel_err,
            colors=colors,
            func=r02_arel_param_rho,
            pl_profile=pl_profile,
            xbins=rho_bins,
            #   p0=[1.02057967, -1.30079264, 1.80129233],
            p0=[2.03427784e-01, 4.04605691e-07, 1.15588788e+07, 0],
            plot_param=plot_param,
            ylabel=r"$a_\mathrm{rel}$",
            xlabel=rho_lab,
            ax_res_xlabel=rho_lab_avg,
            x_avg=rho_mc_avg,
            plot_avg=res_avg,
            ax_res=axs_res[0, 2],
            ylim=[0, 1],
        )

    if not 0:
        p_slope_popt = plot_parameter(
            axs[0, 3],
            dmax_mc,
            p_slope,
            p_slope_err,
            colors=colors,
            func=r02_arel_param,
            pl_profile=pl_profile,
            xbins=dmax_bins,
            p0=[1.50262520e+00,-3.12110331e-07,2.41040614e+07],  # Auger param
            # p0=[1.52326814e+00,-8.41306536e-07,-9.33299796e+07],  # GRAND param
            plot_param=plot_param,
            ylabel=r"Inner Gaussian slope $p_\mathrm{inner}$",
            xlabel=dmax_lab,
            ax_res_xlabel=dmax_lab_avg,
            x_avg=dmax_mc_avg,
            plot_avg=res_avg,
            ax_res=axs_res[0, 3],
            ylim= [1.1, 1.9], # [0.95, 2.55],
            colormap=colormap,
            plot_colormap=show_colormap
        )
    else:
        p_slope_popt = plot_parameter(
            axs[0, 3],
            rho_mc,
            p_slope,
            p_slope_err,
            colors=colors,
            func=None,
            pl_profile=pl_profile,
            xbins=rho_bins,
            p0=[-0.75262595, 1.98703018],
            plot_param=plot_param,
            ylabel=r"$p_slope$",
            xlabel=rho_lab,
            ax_res_xlabel=rho_lab_avg,
            x_avg=rho_mc_avg,
            plot_avg=res_avg,
            ax_res=axs_res[1, 3]
        )

    if np.all(slope == 5):
        axs[1, 2].axhline(5, color="C2")
        plot_parameter(
            axs[1, 2],
            dmax_mc,
            slope,
            np.zeros_like(slope),
            ylabel=r"Sigmoid slope $s$",
            xlabel=dmax_lab,
            ylim=[0, 25],
        )
        axs_res[1, 2].axhline(0)
    else:
        plot_parameter(
            axs[1, 2], dmax_mc, slope, slope_err, ylabel=r"slope", xlabel=dmax_lab,
            ylim=[0, 12],
            colormap=colormap,
            plot_colormap=show_colormap
        )

    [ax.grid() for ax in axs.flatten()]
    [ax.set_xlim(0, 180) for ax in axs.flatten()]
    # [ax.set_xlim(0, 230) for ax in axs.flatten()]
    # [ax.legend() for ax in axs.flatten()]
    # axs[1, 0].set_ylim(0, 800)
    # axs[0, 1].set_ylim(0.5, 1)
    # axs[0, 2].set_ylim(0.7, 1)
    fig.tight_layout()
    fig_res.tight_layout()

    if plot_param:
        [ax.grid() for ax in axs_res.flatten()]
        [ax.set_ylim(-0.3, 0.3) for ax in axs_res.flatten()]
        [ax.legend(fontsize=12) for ax in axs_res.flatten()]

        axs_res[0, 0].set_ylabel(r"$(x_\mathrm{par} - x) / x_\mathrm{par}$")
        axs_res[1, 0].set_ylabel(r"$(x_\mathrm{par} - x) / x_\mathrm{par}$")

        fig_res.tight_layout()
        if para.save:
            fig_res.savefig("pars_res%s.png" % para.label)
    else:
        plt.close(fig_res)

    if 1:
        # draw_offline_param(axs, helpers.get_fine_xs(dmax_mc), "C1-")
        # draw_hard_param(axs, helpers.get_fine_xs(dmax_mc), "C1-")
        draw_soft_param(
            axs, helpers.get_fine_xs(r0_start),
            helpers.get_fine_xs(dmax_mc), helpers.get_fine_xs(dmax_mc_avg),
            "black",
            label="final param.",
            plot_p=True,
        )
        axs[1, 3].legend(fontsize=15)

    if para.save:
        if not plot_single:
            fig.savefig("pars%s.png" % para.label)
        else:
            plt.close(fig)
            for ax, l in zip(axs.flatten(), ["r0", "r02", "arel", "p_slope", "sig", "p", "s"]):
                fig_tmp = plt.figure(figsize=ssize)
                dummy = fig_tmp.add_axes((0.23, 0.20, 0.76, 0.79))

                ax.remove()
                ax.figure = fig_tmp
                fig_tmp.add_axes(ax)

                ax.set_position(dummy.get_position())
                dummy.remove()

                fig_tmp.savefig("pars_%s%s.png" % (l, para.label))
                plt.close()

    else:
        plt.show()
        plt.close()

    
    ###
    '''NEW PLOT'''
    ###

    # new labels
    xmax_lab = r"$x_\mathrm{max}^\mathrm{MC}$ / $g\cdot cm^{-2}$"
    energy_lab = r"log$(E/\mathrm{GeV})$"

    # new values for plot 2
    para_rho = False
    plot_param = False
    pl_profile = False
    res_avg = False
    use_fitted_errors = True
    new_p = True
    plot_single = False # True: plot each parameter in its own plot, False: grouped plot
    ssize = (45, 20)

    # parameter to plot over
    x_parameter = np.log10(energy) # xmax_mc
    dep_label = energy_lab


    # add new figure(s) to show additional dependencies of fit parameters
        
    if plot_single:
        fig, axs = plt.subplots(2, 4, figsize=ssize)
        fig_res, axs_res = plt.subplots(2, 4, figsize=ssize)

    else:
        fig, axs = plt.subplots(2, 4, figsize=ssize)
        # fig_res, axs_res = plt.subplots(2, 4, figsize=ssize)


    plot_parameter(
        axs[0, 0],
        x_parameter,
        r0,
        np.ones_like(r0),
        r"$r_0$ / m",
        xlabel=dep_label,
        colors=colors,
        colormap=colormap,
    )

    # if plot_param:
    #     axs[0, 0].plot(dmax_mc / 1e3, r0_param, "k.")
    #     axs_res[0, 0].plot(dmax_mc / 1e3, (r0_param - r0) /
    #                        r0_param, "C0o", alpha=0.1)

    sig_popt = plot_parameter(
        axs[1, 0],
        x_parameter,
        sig / np.mean(sig),
        sig_err,
        colors=colors,
        pl_profile=pl_profile,
        plot_param=plot_param,
        ylabel=r"$\sigma / \bar{\sigma}$",
        xlabel=dep_label,
        # ylim=[50, 600],
        colormap=colormap,
    )

    r02_popt = plot_parameter(
        axs[0, 1],
        x_parameter,
        r02 / np.mean(r02),
        r02_err,
        colors=colors,
        pl_profile=pl_profile,
        ylabel=r"$r_{02} / \bar{r_{02}}$ ",
        #  ylabel=r"$r_{02}$ [m]",
        plot_param=plot_param,
        xlabel=dep_label,
        # ylim=[0.1, 1],
        colormap=colormap,
    )

    p_popt = plot_parameter(
        axs[1, 1],
        x_parameter,
        p / np.mean(p),
        p_err,
        colors=colors,
        pl_profile=pl_profile,
        plot_param=plot_param,
        ylabel=r"$b / \bar{b}$",
        xlabel=dep_label,           
        # ylim=[150, 400],
        colormap=colormap,
    )

    arel_popt = plot_parameter(
        axs[0, 2],
        x_parameter,
        arel / np.mean(arel),
        arel_err,
        colors=colors,
        pl_profile=pl_profile,
        plot_param=plot_param,
        ylabel=r"$a_\mathrm{rel} / \bar{a_\mathrm{rel}}$",
        xlabel=dep_label,
        # ylim=[0.1, 0.4],
        colormap=colormap,
    )

    p_slope_popt = plot_parameter(
        axs[0, 3],
        x_parameter,
        p_slope / np.mean(p_slope),
        p_slope_err,
        colors=colors,
        pl_profile=pl_profile,
        plot_param=plot_param,
        ylabel=r"$p_{slope} / \bar{p_{slope}}$",
        xlabel=dep_label,
        # ylim=[0.95, 2.3],
        colormap=colormap,
    )

    if np.all(slope == 5):
        axs[1, 2].axhline(5, color="C2")
        plot_parameter(
            axs[1, 2],
            dmax_mc,
            slope,
            np.zeros_like(slope),
            ylabel=r"slope",
            xlabel=dmax_lab,
            ylim=[0, 25]
        )
        axs_res[1, 2].axhline(0)
    else:
        plot_parameter(
            axs[1, 2], dmax_mc, slope, slope_err, ylabel=r"slope", xlabel=dmax_lab,
            ylim=[0, 12],
            colormap=colormap,
        )

    [ax.grid() for ax in axs.flatten()]
    # [ax.set_xlim(0, 180) for ax in axs.flatten()]
    # [ax.legend() for ax in axs.flatten()]
    # axs[1, 0].set_ylim(0, 800)
    # axs[0, 1].set_ylim(0.5, 1)
    # axs[0, 2].set_ylim(0.7, 1)
    fig.tight_layout()
    fig_res.tight_layout()

    if plot_param:
        [ax.grid() for ax in axs_res.flatten()]
        [ax.set_ylim(-0.3, 0.3) for ax in axs_res.flatten()]
        [ax.legend(fontsize=12) for ax in axs_res.flatten()]

        axs_res[0, 0].set_ylabel(r"$(x_\mathrm{par} - x) / x_\mathrm{par}$")
        axs_res[1, 0].set_ylabel(r"$(x_\mathrm{par} - x) / x_\mathrm{par}$")

        fig_res.tight_layout()
        if para.save:
            fig_res.savefig("pars_res%s.png" % para.label)
    else:
        plt.close(fig_res)

    if para.save:
        if not plot_single:
            fig.savefig("pars_dependencies.png")
        else:
            plt.close(fig)
            for ax, l in zip(axs.flatten(), ["r0", "r02", "arel", "p_slope", "sig", "p", "s"]):
                fig_tmp = plt.figure(figsize=ssize)
                dummy = fig_tmp.add_axes((0.23, 0.20, 0.76, 0.79))

                ax.remove()
                ax.figure = fig_tmp
                fig_tmp.add_axes(ax)

                ax.set_position(dummy.get_position())
                dummy.remove()

                fig_tmp.savefig("pars_dependencies.png")

    else:
        plt.show()


    # sys.exit()
