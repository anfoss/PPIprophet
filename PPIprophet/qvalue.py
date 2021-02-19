import numpy as np
import scipy as sp
import pandas as pd
import math
import scipy.stats
import scipy.special
import multiprocessing
from statsmodels.nonparametric.kde import KDEUnivariate
from collections import namedtuple



def mean_and_std_dev(values):
    return np.mean(values), np.std(values, ddof=1)


def pnorm(stat, stat0):
    """ [P(X>pi, mu, sigma) for pi in pvalues] for normal distributed stat with
    expectation value mu and std deviation sigma """

    mu, sigma = mean_and_std_dev(stat0)

    args = (stat - mu) / sigma
    return 1-(0.5 * (1.0 + scipy.special.erf(args / np.sqrt(2.0))))


def pemp(stat, stat0):
    """ Computes empirical values identically to bioconductor/qvalue empPvals """

    assert len(stat0) > 0
    assert len(stat) > 0

    stat = np.array(stat)
    stat0 = np.array(stat0)

    m = len(stat)
    m0 = len(stat0)

    statc = np.concatenate((stat, stat0))
    v = np.array([True] * m + [False] * m0)
    perm = np.argsort(-statc, kind="mergesort")  # reversed sort, mergesort is stable
    v = v[perm]

    u = np.where(v)[0]
    p = (u - np.arange(m)) / float(m0)

    # ranks can be fractional, we round down to the next integer, ranking returns values starting
    # with 1, not 0:
    ranks = np.floor(scipy.stats.rankdata(-stat)).astype(int) - 1
    p = p[ranks]
    p[p <= 1.0 / m0] = 1.0 / m0

    return p


def pi0est(p_values, lambda_ = np.arange(0.05,1.0,0.05), pi0_method = "smoother", smooth_df = 3, smooth_log_pi0 = False):
    """ Estimate pi0 according to bioconductor/qvalue """

    # Compare to bioconductor/qvalue reference implementation
    # import rpy2
    # import rpy2.robjects as robjects
    # from rpy2.robjects import pandas2ri
    # pandas2ri.activate()

    # smoothspline=robjects.r('smooth.spline')
    # predict=robjects.r('predict')

    p = np.array(p_values)

    rm_na = np.isfinite(p)
    p = p[rm_na]
    m = len(p)
    ll = 1
    if isinstance(lambda_, np.ndarray ):
        ll = len(lambda_)
        lambda_ = np.sort(lambda_)

    if (min(p) < 0 or max(p) > 1):
        raise ValueError("p-values not in valid range [0,1].")
    elif (ll > 1 and ll < 4):
        raise ValueError("If lambda_ is not predefined (one value), at least four data points are required.")
    elif (np.min(lambda_) < 0 or np.max(lambda_) >= 1):
        raise ValueError("Lambda must be within [0,1)")

    if (ll == 1):
        pi0 = np.mean(p >= lambda_)/(1 - lambda_)
        pi0_lambda = pi0
        pi0 = np.minimum(pi0, 1)
        pi0Smooth = False
    else:
        pi0 = []
        for l in lambda_:
            pi0.append(np.mean(p >= l)/(1 - l))
        pi0_lambda = pi0

        if (pi0_method == "smoother"):
            if smooth_log_pi0:
                pi0 = np.log(pi0)
                spi0 = sp.interpolate.UnivariateSpline(lambda_, pi0, k=smooth_df)
                pi0Smooth = np.exp(spi0(lambda_))
                # spi0 = smoothspline(lambda_, pi0, df = smooth_df) # R reference function
                # pi0Smooth = np.exp(predict(spi0, x = lambda_).rx2('y')) # R reference function
            else:
                spi0 = sp.interpolate.UnivariateSpline(lambda_, pi0, k=smooth_df)
                pi0Smooth = spi0(lambda_)
                # spi0 = smoothspline(lambda_, pi0, df = smooth_df) # R reference function
                # pi0Smooth = predict(spi0, x = lambda_).rx2('y')  # R reference function
            pi0 = np.minimum(pi0Smooth[ll-1],1)
        elif (pi0_method == "bootstrap"):
            minpi0 = np.percentile(pi0,0.1)
            W = []
            for l in lambda_:
                W.append(np.sum(p >= l))
            mse = (np.array(W) / (np.power(m,2) * np.power((1 - lambda_),2))) * (1 - np.array(W) / m) + np.power((pi0 - minpi0),2)
            pi0 = np.minimum(pi0[np.argmin(mse)],1)
            pi0Smooth = False
        else:
            raise ValueError("pi0_method must be one of 'smoother' or 'bootstrap'.")
    if (pi0<=0):
        raise ValueError("The estimated pi0 <= 0. Check that you have valid p-values or use a different range of lambda.")

    return {'pi0': pi0, 'pi0_lambda': pi0_lambda, 'lambda_': lambda_, 'pi0_smooth': pi0Smooth}

def qvalue(p_values, pi0, pfdr = False):
    p = np.array(p_values)

    qvals_out = p
    rm_na = np.isfinite(p)
    p = p[rm_na]

    if (min(p) < 0 or max(p) > 1):
        raise ValueError("p-values not in valid range [0,1].")
    elif (pi0 < 0 or pi0 > 1):
        raise ValueError("pi0 not in valid range [0,1].")

    m = len(p)
    u = np.argsort(p)
    v = scipy.stats.rankdata(p,"max")

    if pfdr:
        qvals = (pi0 * m * p) / (v * (1 - np.power((1 - p), m)))
    else:
        qvals = (pi0 * m * p) / v

    qvals[u[m-1]] = np.minimum(qvals[u[m-1]], 1)
    for i in list(reversed(range(0,m-2,1))):
        qvals[u[i]] = np.minimum(qvals[u[i]], qvals[u[i + 1]])

    qvals_out[rm_na] = qvals
    return qvals_out


def bw_nrd0(x):
    if len(x) < 2:
        raise ValueError("bandwidth estimation requires at least two data points.")

    hi = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    lo = min(hi, iqr/1.34)
    lo = lo or hi or abs(x[0]) or 1

    return 0.9 * lo *len(x)**-0.2


def lfdr(p_values, pi0, trunc = True, monotone = True, transf = "probit", adj = 1.5, eps = np.power(10.0,-8)):
    """ Estimate local FDR / posterior error probability from p-values according to bioconductor/qvalue """
    p = np.array(p_values)

    # Compare to bioconductor/qvalue reference implementation
    # import rpy2
    # import rpy2.robjects as robjects
    # from rpy2.robjects import pandas2ri
    # pandas2ri.activate()

    # density=robjects.r('density')
    # smoothspline=robjects.r('smooth.spline')
    # predict=robjects.r('predict')

    # Check inputs
    lfdr_out = p
    rm_na = np.isfinite(p)
    p = p[rm_na]

    if (min(p) < 0 or max(p) > 1):
        raise ValueError("p-values not in valid range [0,1].")
    elif (pi0 < 0 or pi0 > 1):
        raise ValueError("pi0 not in valid range [0,1].")

    # Local FDR method for both probit and logit transformations
    if (transf == "probit"):
        p = np.maximum(p, eps)
        p = np.minimum(p, 1-eps)
        x = scipy.stats.norm.ppf(p, loc=0, scale=1)

        # R-like implementation
        bw = bw_nrd0(x)
        myd = KDEUnivariate(x)
        myd.fit(bw=adj*bw, gridsize = 512)
        splinefit = sp.interpolate.splrep(myd.support, myd.density)
        y = sp.interpolate.splev(x, splinefit)
        # myd = density(x, adjust = 1.5) # R reference function
        # mys = smoothspline(x = myd.rx2('x'), y = myd.rx2('y')) # R reference function
        # y = predict(mys, x).rx2('y') # R reference function

        lfdr = pi0 * scipy.stats.norm.pdf(x) / y
    elif (transf == "logit"):
        x = np.log((p + eps) / (1 - p + eps))

        # R-like implementation
        bw = bw_nrd0(x)
        myd = KDEUnivariate(x)
        myd.fit(bw=adj*bw, gridsize = 512)

        splinefit = sp.interpolate.splrep(myd.support, myd.density)
        y = sp.interpolate.splev(x, splinefit)
        # myd = density(x, adjust = 1.5) # R reference function
        # mys = smoothspline(x = myd.rx2('x'), y = myd.rx2('y')) # R reference function
        # y = predict(mys, x).rx2('y') # R reference function

        dx = np.exp(x) / np.power((1 + np.exp(x)),2)
        lfdr = (pi0 * dx) / y
    else:
        raise ValueError("Invalid local FDR method.")

    if (trunc):
        lfdr[lfdr > 1] = 1
    if (monotone):
        lfdr = lfdr[p.ravel().argsort()]
        for i in range(1,len(x)):
            if (lfdr[i] < lfdr[i - 1]):
                lfdr[i] = lfdr[i - 1]
        lfdr = lfdr[scipy.stats.rankdata(p,"min")-1]

    lfdr_out[rm_na] = lfdr
    return lfdr_out


def final_err_table(df, num_cut_offs=51):
    """ Create artificial cutoff sample points from given range of cutoff
    values in df, number of sample points is 'num_cut_offs' """

    cutoffs = df.cutoff.values
    min_ = min(cutoffs)
    max_ = max(cutoffs)
    # extend max_ and min_ by 5 % of full range
    margin = (max_ - min_) * 0.05
    sampled_cutoffs = np.linspace(min_ - margin, max_ + margin, num_cut_offs, dtype=np.float32)

    # find best matching row index for each sampled cut off:
    ix = find_nearest_matches(np.float32(df.cutoff.values), sampled_cutoffs)

    # create sub dataframe:
    sampled_df = df.iloc[ix].copy()
    sampled_df.cutoff = sampled_cutoffs
    # remove 'old' index from input df:
    sampled_df.reset_index(inplace=True, drop=True)

    return sampled_df


def summary_err_table(df, qvalues=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
    """ Summary error table for some typical q-values """

    # qvalues = qvalues
    # find best matching fows in df for given qvalues:
    ix = find_nearest_matches(np.float32(df.qvalue.values), qvalues)
    # extract sub table
    df_sub = df.iloc[ix].copy()
    # remove duplicate hits, mark them with None / NAN:
    for i_sub, (i0, i1) in enumerate(zip(ix, ix[1:])):
        if i1 == i0:
            df_sub.iloc[i_sub + 1, :] = None
    # attach q values column
    df_sub.qvalue = qvalues
    # remove old index from original df:
    df_sub.reset_index(inplace=True, drop=True)
    return df_sub[['qvalue','pvalue','svalue','pep','fdr','fnr','fpr','tp','tn','fp','fn','cutoff']]


def error_statistics(target_scores, decoy_scores, parametric=False, pfdr=0.1, pi0_lambda=False, pi0_method = "smoother", pi0_smooth_df = 3, pi0_smooth_log_pi0 = False, compute_lfdr = True, lfdr_trunc = True, lfdr_monotone = True, lfdr_transf = "probit", lfdr_adj = 1.5, lfdr_eps = np.power(10.0,-8)):

    target_scores = np.sort(target_scores[~np.isnan(target_scores)])
    decoy_scores = np.sort(decoy_scores[~np.isnan(decoy_scores)])

    # compute p-values using decoy scores
    if parametric:
        # parametric
        target_pvalues = pnorm(target_scores, decoy_scores)
    else:
        # non-parametric
        target_pvalues = pemp(target_scores, decoy_scores)

    # estimate pi0
    pi0 = pi0est(target_pvalues, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0)

    # compute q-value
    target_qvalues = qvalue(target_pvalues, pi0['pi0'], pfdr)


    # generate main statistics table
    error_stat = pd.DataFrame({'cutoff': target_scores, 'pvalue': target_pvalues, 'qvalue': target_qvalues})

    # compute lfdr / PEP
    if compute_lfdr:
        error_stat['pep'] = lfdr(target_pvalues, pi0['pi0'], lfdr_trunc, lfdr_monotone, lfdr_transf, lfdr_adj, lfdr_eps)

    return error_stat


def find_cutoff(tt_scores, td_scores, cutoff_fdr, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0):
    """ Finds cut off target score for specified false discovery rate fdr """

    error_stat, pi0 = error_statistics(tt_scores, td_scores, parametric, pfdr, pi0_lambda, pi0_method, pi0_smooth_df, pi0_smooth_log_pi0, False)
    i0 = (error_stat.qvalue - cutoff_fdr).abs().idxmin()
    cutoff = error_stat.iloc[i0]["cutoff"]
    return cutoff
