import numpy as np


def figure_of_merit(dist1, dist2):
    # calculate separation of two (Gaussian) distributions
    return np.abs(np.mean(dist1) - np.mean(dist2)) / \
        np.sqrt(np.std(dist1) ** 2 + np.std(dist2) ** 2)


def binomial_proportion(nsel, ntot, coverage=0.68):
    """
    Copied from pyik.mumpyext (original author HD)
    
    Calculate a binomial proportion (e.g. efficiency of a selection) and its confidence interval.

    Parameters
    ----------
    nsel: array-like
      Number of selected events.
    ntot: array-like
      Total number of events.
    coverage: float (optional)
      Requested fractional coverage of interval (default: 0.68).

    Returns
    -------
    p: array of dtype float
      Binomial fraction.
    dpl: array of dtype float
      Lower uncertainty delta (p - pLow).
    dpu: array of dtype float
      Upper uncertainty delta (pUp - p).

    Examples
    --------
    >>> p,dpl,dpu = binomial_proportion(50,100,0.68)
    >>> print "%.4f %.4f %.4f" % (p,dpl,dpu)
    0.5000 0.0495 0.0495
    >>> abs(np.sqrt(0.5*(1.0-0.5)/100.0)-0.5*(dpl+dpu)) < 1e-3
    True

    Notes
    -----
    The confidence interval is approximate and uses the score method
    of Wilson. It is based on the log-likelihood profile and can
    undercover the true interval, but the coverage is on average
    closer to the nominal coverage than the exact Clopper-Pearson
    interval. It is impossible to achieve perfect nominal coverage
    as a consequence of the discreteness of the data.
    """

    from scipy.stats import norm

    z = norm().ppf(0.5 + 0.5 * coverage)
    z2 = z * z
    p = np.asarray(nsel, dtype=float) / ntot
    div = 1.0 + z2 / ntot
    pm = (p + z2 / (2 * ntot))
    dp = z * np.sqrt(p * (1.0 - p) / ntot + z2 / (4 * ntot * ntot))
    pl = (pm - dp) / div
    pu = (pm + dp) / div

    return p, p - pl, pu - p


def calculate_efficiency_and_error(selected_events, total_events):
    """ 
    Calculate the efficiency and associated 1-sigma confidence interval assuming
    binominal statistics.
    
    Input arrays can be of arbitrary shape. Out put arrays will have the same shape.
    perr will have an extra dimension for lower and upper uncertainty.

    Parameters
    ----------

    selected_events : array
        Number of selected events.
    
    total_events : array
        Number of total events.

    Returns
    -------

    p : array
        Efficency (selected_events / total_events)

    perr : array
        Uncertainty (1-sigma convidence interval), asymmetric
    """
    
    p, pelow, peup = \
        np.zeros(total_events.shape), np.zeros(total_events.shape), \
        np.zeros(total_events.shape)
    
    not_null = total_events != 0
    # print(total_events, not_null)
    p[not_null], pelow[not_null], peup[not_null] = \
        binomial_proportion(selected_events[not_null], total_events[not_null])
    
    # set unreasonably small errors to 0
    # most applied for upper limit when efficiency hits 1 and can't be higher
    for i in range(len(peup)):
        for j in range(len(peup[i])):
            if peup[i, j] < 1e-10:
                peup[i, j] = 0

    perr = np.array([pelow, peup])
    return p, perr


def get_binned_data(x, y, x_bins, return_bins=False, skip_empty_bins=False):
    if y.ndim == 1:
        # binning[0] -> bin entries, binning[1] -> bin center, binning[2] -> mean value of each bin,  binning[3] -> std value of each bin

        # n[0]: Array of entries in each bin, n[1]: bin edges
        n, bins = np.histogram(x, bins=x_bins)

        # sy: Array of sum(y) for each bin
        sy, _ = np.histogram(x, bins=x_bins, weights=y)
        sy2, _ = np.histogram(x, bins=x_bins, weights=y ** 2)

        # checking for bad bins
        y_mean_binned = np.zeros(sy.shape)
        y_std_binned = np.zeros(sy.shape)
        if isinstance(skip_empty_bins, int):
            null_mask = np.all([sy != 0, n >= skip_empty_bins], axis=0)

        # Binned mean values of y
        y_mean_binned[null_mask] = sy[null_mask] / n[null_mask]

        # Binned std values of y
        y_var_binned = \
            np.abs(sy2[null_mask] / n[null_mask] -
                   y_mean_binned[null_mask] ** 2).astype(float)

        # was needed to pull the sqrt out of the previous line to avoid an error. no idea why
        y_std_binned[null_mask] = np.sqrt(y_var_binned)

        # bin value (center)
        bins_center = bins[:-1] + (bins[1:] - bins[:-1]) / 2.

        if skip_empty_bins:
            return_mask = null_mask
        else:
            return_mask = np.full_like(n, True, dtype=bool)

        if return_bins:
            return n[return_mask], bins_center[return_mask], y_mean_binned[return_mask], y_std_binned[return_mask], bins
        else:
            return n[return_mask], bins_center[return_mask], y_mean_binned[return_mask], y_std_binned[return_mask]

    elif y.ndim == 2:
        # binning[0] -> bin entries, binning[1] -> bin center, binning[2] -> mean value of each bin,  binning[3] -> std value of each bin
        N, bins_center, y_mean_binned, y_std_binned = [], [], [], []
        for idx, elem in enumerate(y):
            # n[0]: Array of entries in each bin, n[1]: bin edges
            n, bins = np.histogram(x, bins=x_bins)
            # sy: Array of sum(y) for each bin
            sy, _ = np.histogram(x, bins=x_bins, weights=elem)
            sy2, _ = np.histogram(x, bins=x_bins, weights=elem ** 2)

            null_mask = (sy != 0)  # checking for bad bins
            # Binned mean values of y
            y_mean_binned.append(np.where(null_mask, sy / n, 0))
            # Binned std values of y
            y_std_binned.append(np.where(null_mask, np.sqrt(
                np.abs(sy2 / n - np.power(np.where(null_mask, sy / n, 0), 2.))), 0))
            # bin value (center)
            bins_center.append(bins[:-1] + (bins[1:] - bins[:-1]) / 2.)
            N.append(n)

        return np.array(N), np.array(bins_center)[0], np.array(y_mean_binned), np.array(y_std_binned)

    else:
        print("Wrong dimension of y")


def get_quantiles(x, y, xbins, quantiles=[0.68]):

    xc = None
    if isinstance(xbins, int):
        _, xbins = np.histogram(x, xbins)
        xc = xbins[:-1] + (xbins[1:] - xbins[:-1]) / 2

    quants = np.full((len(xbins) - 1, len(quantiles)), np.nan)

    for idx in range(len(xbins) - 1):
        mask = np.all([x >= xbins[idx], x < xbins[idx+1]], axis=0)
        if not np.any(mask):
            continue

        for qdx, q in enumerate(quantiles):
            quants[idx, qdx] = np.quantile(y[mask], q)

    quants = np.squeeze(quants)

    if xc is not None:
        if quants.ndim == 1:
            return xc, quants
        else:
            # append [xc] and [arr1, arr2, ...] together
            return [xc] + np.hsplit(quants, len(quantiles))
    else:
        if quants.ndim == 1:
            return quants
        else:
            return np.hsplit(quants, len(quantiles))


def normed_deviation(arr1, arr2):
    return (arr1 - arr2) / (0.5 * (arr1 + arr2))


def normed_deviation2(arr_true, arr_test):
    return (arr_true - arr_test) / arr_true


def bootstrapping(arr, n=1000, p=1, pri=True):
    nominal_mean = np.mean(arr)
    nominal_std = np.std(arr)
    means = np.zeros(n)
    stds = np.zeros(n)
    for idx in range(n):
        arr_sup = np.random.choice(arr, int(len(arr) * p))
        means[idx] = np.mean(arr_sup)
        stds[idx] = np.std(arr_sup)
    uncert_std = np.std(stds)
    uncert_mean = np.std(means)
    if pri:
        print("Mean: {} +- {}".format(nominal_mean, uncert_mean))
        print("Std: {} +- {}".format(nominal_std, uncert_std))
    return nominal_mean, uncert_mean, nominal_std, uncert_std


def selection_efficiency_purity(
        obs_1, obs_2, efficiency=np.arange(1, 100, 1)):
    """
    For two distributions calculate the efficiency and purity with which one 
    can select the elements of the first distribution w.r.t. the second. 
    """

    # sort for easy calculation => 'counting from the left side'
    obs_1 = np.sort(obs_1)

    losses = []
    purities = []
    contaminations = []

    print(np.mean(obs_1), np.mean(obs_2))

    # calculate loss, purity, contamination from the right or left side
    if np.mean(obs_1) < np.mean(obs_2):
        for eff in efficiency:
            # since obs_1 is sorted
            t_bin = int(eff / 100 * len(obs_1))
            t_value = obs_1[t_bin]

            n_obs_2 = np.sum(obs_2 < t_value)
            n_tot = n_obs_2 + t_bin  # since sorted t_bin = n_obs_1

            losses.append(100 - eff)
            purities.append(100 - (n_obs_2 / n_tot) * 100)
            contaminations.append(n_obs_2 / n_tot * 100)

    else:
        # flip = 'counting from right other side'
        obs_1 = np.flip(obs_1)
        for eff in efficiency:
            # since obs_1 is sorted
            t_bin = int(eff / 100 * len(obs_1))
            t_value = obs_1[t_bin]

            n_obs_2 = np.sum(obs_2 > t_value)
            n_tot = n_obs_2 + t_bin  # since sorted t_bin = n_obs_1

            losses.append(100 - eff)
            purities.append(100 - (n_obs_2 / n_tot) * 100)
            contaminations.append(n_obs_2 / n_tot * 100)

    return np.array(losses), np.array(purities), np.array(contaminations)


def determine_fischer(c1_x, c1_y, c2_x, c2_y):
    """ 
    Simple fisher discriminant analysis.

    Taken from https://scipy.github.io/old-wiki/pages/Cookbook/LinearClassification.html

    """
    c1_data = np.array([c1_x, c1_y]).T
    c2_data = np.array([c2_x, c2_y]).T

    c1_mean = np.mean(c1_data, axis=0)
    c2_mean = np.mean(c2_data, axis=0)

    Sw = np.dot((c1_data - c1_mean).T, (c1_data - c1_mean)) + \
        np.dot((c2_data - c2_mean).T, (c2_data - c2_mean))

    # calculate weights which maximize linear separation
    w = np.dot(np.linalg.inv(Sw), (c2_mean - c1_mean))

    # projection of classes on 1D space
    projected_c1_data = np.dot(c1_data, w)
    projected_c2_data = np.dot(c2_data, w)

    return projected_c1_data, projected_c2_data, Sw, w
