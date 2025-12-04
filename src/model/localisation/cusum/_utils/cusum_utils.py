#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:49:24 2025

@author: ellasteins
"""

# %% imports


import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()

# matplotlib.rc('text', usetex = True)
# params = {'text.latex.preamble': r'\usepackage{amsmath}'}
# plt.rcParams.update(params)


from statsmodels.stats.stattools import durbin_watson


# %% SUMMARY CUSUM (combing method adn and corr)


def f_CUSUM_serializable(
    df, pipe_id, ground_truth, h_adn=300, h_corr=250, return_data: bool = True
):
    """
    Works with numeric index (in seconds) so it can be parallelized
    CONVERT INDEX TIMEDELTA TO SECONDS!!!
    """

    true_leak_start = ground_truth[pipe_id][0]  # leak start, seconds since 2019-01-01
    true_leak_fix = ground_truth[pipe_id][1]  # leak fix, seconds since 2019-01-01
    DMA = ground_truth[pipe_id][
        2
    ]  # DMA of the leak -> used to select the sensor combinations that are tested

    # for each leak, CUSUM starts 2 weeks before (unless its less than two weeks after start of data record) and ends when the leak is fixed
    if true_leak_start >= 14 * 24 * 60 * 60:  # D*H*M*S
        start_time = true_leak_start - 14 * 24 * 60 * 60
    else:
        start_time = 0

    end_time = true_leak_fix

    df = df[df.index >= start_time]
    df = df[df.index <= end_time]

    if DMA == "A" or DMA == "B":
        # no irregularities -> method adn (CUSUM) will be used
        leak_det, det, C = f_CUSUM_adn(df, h_thr_alter=h_adn)  # adn method
    else:
        # irregularities -> method corr (CUSUM) will be used (which is method adn + decorrelation step to minimize the random fluctuations)
        leak_det, det, C = f_CUSUM_corr(df, h_thr_alter=h_corr)  # corr method

    if det == 0:
        detection = "FN"
    else:
        TTD_leak = leak_det - true_leak_start
        if TTD_leak >= 0:
            detection = TTD_leak
        else:
            detection = "FP"

    if not return_data:
        return detection

    df_C = pd.DataFrame(index=df.index, data=C)

    return (detection, df_C)


def f_CUSUM(
    df,
    pipe_id,
    ground_truth,
    h_adn=300,
    h_corr=250,
    return_data: bool = True,
):

    true_leak_start = ground_truth[pipe_id][0]  # leak start
    true_leak_fix = ground_truth[pipe_id][1]  # leak fix
    DMA = ground_truth[pipe_id][
        2
    ]  # DMA of the leak -> used to select the sensor combinations that are tested

    # for each leak, CUSUM starts 2 weeks before (unless its less than two weeks after start of data record) and ends when the leak is fixed
    if pd.Timestamp(true_leak_start) >= pd.Timestamp(
        "2019-01-14 00:00:00"
    ):  # this is for 2019, might need to change if we use another year
        start_time = pd.Timestamp(true_leak_start) - pd.Timedelta("14 days")
    else:
        start_time = pd.Timestamp("2019-01-01 00:00:00")

    end_time = pd.Timestamp(true_leak_fix)
    df = df[df.index >= start_time]
    df = df[df.index <= end_time]

    if DMA == "A" or DMA == "B":
        # no irregularities -> method adn (CUSUM) will be used
        leak_det, det, C = f_CUSUM_adn(df, h_thr_alter=h_adn)  # adn method
    else:
        # irregularities -> method corr (CUSUM) will be used (which is method adn + decorrelation step to minimize the random fluctuations)
        leak_det, det, C = f_CUSUM_corr(df, h_thr_alter=h_corr)  # corr method

    if det == 0:
        detection = "FN"
    else:
        TTD_leak = leak_det - pd.Timestamp(true_leak_start)
        TTD_leak_seconds = TTD_leak.total_seconds()
        if TTD_leak_seconds >= 0:
            detection = TTD_leak
        else:
            detection = "FP"

    if not return_data:
        return detection

    df_C = pd.DataFrame(index=df.index, data=C)

    return (detection, df_C)


# %% Plotting the CUSUM statistic


def f_plot_C(df_C, det_leak, pipe_id, ground_truth, h_adn=300, h_corr=250):

    true_leak_start = ground_truth[pipe_id][0]
    true_leak_fix = ground_truth[pipe_id][1]

    DMA = ground_truth[pipe_id][2]
    if DMA == "DMA_C":
        method = "corr"
        h = h_corr
    else:
        method = "adn"
        h = h_adn

    if det_leak == "FP" or det_leak == "FN":
        if det_leak == "FP":
            end_time = pd.Timestamp(true_leak_start) + pd.Timedelta(hours=3)
        else:
            end_time = pd.Timestamp(true_leak_fix)
        A = 0
    else:
        end_time = (
            pd.Timestamp(true_leak_start)
            + pd.Timedelta(det_leak)
            + pd.Timedelta(hours=3)
        )
        A = 1

    filtered_df_C = df_C[df_C.index <= end_time]

    plt.rcParams["text.usetex"] = False
    f, ax = plt.subplots(1, sharex=True, sharey=True, figsize=(18, 7))
    plt.title("Leak in " + str(pipe_id) + ", with " + method + " method", fontsize=16)
    # plt.plot(filtered_df_C, linestyle="-", color="C2")
    ax.axvline(
        pd.Timestamp(true_leak_start),
        linestyle="--",
        color="gray",
        label="true leak start",
        lw=2,
    )
    if A == 1:
        ax.axvline(
            pd.Timestamp(true_leak_start) + pd.Timedelta(det_leak),
            linestyle="--",
            color="C1",
            label="leak detection",
            lw=2,
        )
        plt.plot(filtered_df_C, lw=2, linestyle="-", color="C2")
    else:
        plt.plot(filtered_df_C, linestyle="--", color="C2")
    ax.axhline(h, linestyle="-", color="red", lw=0.5, alpha=0.8, label="threshold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("time [date]", fontsize=16)
    plt.ylabel("$C^+$ [-]", fontsize=16)
    plt.legend(fontsize="x-large")
    plt.show()

    return f


# %% CUSUM method 1: adn


def f_CUSUM_adn(df, h_thr_alter=300, IC_ARL=30, delta_0=0.7, m=2):
    """Adaptive and non-parametric CUSUM per Liu, Tsang, and Zhang.
    Adaptive nonparametric CUSUSM scheme for detecting unknown shifts in location. 2014.

    df        :  data to analyze
    IC_ARL    :  IC ARL, choose IC_ARL = {200, 300, 400, 500, 800, 1000}
    h_thr     :  threshold. based in IC ARL. For Y_t = (Rst_t + Rst_(t-1))/2 and delta_0 = 0.7, h_thr in Table 3 for different IC_ARLs
    Y_t       :  one-step ahead estimate of shift in R_t, either EWMA-type or Shewart-type forecast, recommendation if shift magnitude is unknown: Y_t = (Rst_t + Rst_(t-1))/2
    delta_0   :  minimum magnitude of interest for early detection, recommendation if shift magnitude is unknown: delta_0 = 0.7
    Rst_t     :  standardised sequential rank
    R_t       :  sequential rank
    h(k)      :  operating fct, which denotes the control limit (depending on chosen IC_ARL) with reference value k, k = [0.1, 0.866]

    ---
    df_cs:  pd.DataFrame containing cusum-values for each df column
    """

    def f_hk(IC_ARL, k):
        a = np.zeros(9)  # coeff of polynomial
        if IC_ARL == 30:
            a = np.array(
                [
                    13.54844243,
                    -278.74909693,
                    953.99916442,
                    -1436.40494624,
                    1161.01248598,
                    -535.16425988,
                    141.95524256,
                    -24.57009576,
                    5.02083018,
                ]
            )
            h_k = 0
        for i in range(a.shape[0]):
            h_k = h_k + a[i] * k ** (a.shape[0] - 1 - i)
        return h_k

    def f_Yt(Rst_t, Rst_tminus1):
        Y_t = (Rst_t + Rst_tminus1) / 2
        return Y_t

    def f_Rst(x):
        t = x.size
        x_t = x[-1]
        R_t = np.sum(x_t >= x)
        E = (t + 1) / 2
        V = ((t + 1) * (t - 1)) / 12
        Rst_t = (R_t - E) / np.sqrt(V)
        return Rst_t

    for i, col in enumerate(df.columns):
        traj_ = df[col].copy()

    X = traj_.to_numpy()

    S = np.zeros(df.shape)
    S[0, :] = 0
    Rst = np.zeros(df.shape[0])
    Yt = 0

    for t in range(m + 1, df.shape[0]):
        Rst[t] = f_Rst(X[0 : t + 1])
        Yt = f_Yt(Rst[t], Rst[t - 1])
        delta_t = max(Yt, delta_0)
        k = delta_t / 2
        hk = f_hk(IC_ARL, k)
        z = S[t - 1, :] + ((Rst[t] - k) / hk)
        S[t, :] = max(0, z)

    df_S = pd.DataFrame(S, columns=df.columns, index=df.index)
    leak_det = pd.Series(dtype=object)

    for i, pipe in enumerate(df_S):
        hthr = h_thr_alter
        if any(df_S[pipe] > hthr):
            leak_det[pipe] = df_S.index[(df_S[pipe] > hthr).values][0]
            det = 1
            leak_det = leak_det[0]
        else:
            det = 0

    return (leak_det, det, S)


# %% CUSUMmethod 2: corr


def f_CUSUM_corr(df, h_thr_alter=250, m=200, bmax=10, delta_0=0.7, IC_ARL=30):
    """
    combination of:

        Liu. nonparametric and adaptive

        with

    """
    """ Li and Qiu. A general charting scheme for monitoring serially correlated
        data with short-memory dependence and nonparametric distributions. 2020.

        df             :  data to analyze
        m              : number of reference data points at beginning for decorrelation
        dw_stat        : Durbin-Watson statistic of decorrelated initial samples -> should be close to 2
        ARL=           :  IC ARL, here fixed to IC_ARL = 30; otherwise different coeffs for interpolation
        h_thr_alter    :  empirical threshold
        Y_t            :  one-step ahead estimate of shift in R_t, either EWMA-type or Shewart-type forecast, recommendation if shift magnitude is unknown: Y_t = (Rst_t + Rst_(t-1))/2
        delta_0        :  minimum magnitude of interest for early detection, recommendation if shift magnitude is unknown: delta_0 = 0.7
        Rst_t          :  standardised sequential rank
        R_t            :  sequential rank
        h(k)           :  operating fct, which denotes the control limit (depending on chosen IC_ARL) with reference value k, k = [0.1, 0.866]
        X_star         : decorrelized data points
        bmax           : max number of time steps, for which two samples are supposed to be correlated

        ---
        df_cs:  pd.DataFrame containing cusum-values for each df column
        """

    def f_initial_decoorelation(X_IC, m, bmax):
        mu0 = 1 / m * np.sum(X_IC)
        gamma0 = np.zeros(bmax + 1)
        for i in range(bmax + 1):
            factor = 0
            for j in range(m - i):
                factor = factor + (X_IC[j + i, 0] - mu0) * (X_IC[j, 0] - mu0)
            gamma0[i] = 1 / (m - i) * factor

        cov = np.zeros([bmax + 1, bmax + 1])
        for i in range(bmax + 1):
            for j in range(bmax + 1):
                cov[i, j] = gamma0[np.abs(i - j)]

        X_star = np.zeros(X_IC.shape[0])

        for i in range(X_IC.shape[0]):
            if i == 0:
                X_star[i] = (X_IC[i] - mu0) / np.sqrt(gamma0[0])
                b = 1
            else:
                sigma = gamma0[1 : b + 1][::-1].T
                cov_i = cov[0:b, 0:b]
                cov_inv = np.linalg.inv(cov_i)
                d1 = np.sqrt(
                    np.abs(gamma0[0] - np.dot(np.dot(sigma.T, cov_inv), sigma))
                )  # ?
                e = X_IC[i - b : i] - mu0
                factor = np.dot(np.dot(sigma.T, cov_inv), e)
                X_star[i] = (X_IC[i] - mu0 - factor * 1) / d1
                b = min(b + 1, bmax)
        dw_stat = durbin_watson(X_star)  # should be close to 2
        return (X_star, mu0, gamma0, dw_stat)

    def f_Xtstar(Xstar, Xt, mu, gamma, Tau, i0, m):
        tau_length = Tau
        if i0 == m:
            Xt_star = (Xt - mu) * 1 / np.sqrt(gamma[0])
        else:
            if tau_length == 0:
                Xt_star = (Xt - mu) * 1 / np.sqrt(gamma[0])
            else:
                cov = np.zeros([bmax + 1, bmax + 1])
                for i in range(bmax + 1):
                    for j in range(bmax + 1):
                        cov[i, j] = gamma[np.abs(i - j)]
                cov_i = cov[0:tau_length, 0:tau_length]
                cov_inv = np.linalg.inv(cov_i)
                sigma = gamma[1 : tau_length + 1][::-1].T
                e = Xstar[i0 - tau_length : i0] - mu
                d1 = np.sqrt(np.abs(gamma[0] - np.dot(np.dot(sigma.T, cov_inv), sigma)))
                factor = np.dot(np.dot(sigma.T, cov_inv), e)
                Xt_star = (Xt - mu - factor * 1) / d1
        return Xt_star

    def f_moments(Xstar, Xt_star, mu, gamma, i, m0):
        n = i - m0
        mu1 = 1 / (m0 + n) * Xt_star + (m0 + n - 1) / (m0 + n) * mu
        gamma1 = np.zeros(bmax + 1)
        for j in range(bmax + 1):
            gamma1[j] = (
                1 / (m0 + n - j) * (Xt_star - mu1) * (Xstar[i - j] - mu1)
                + (m0 + n - j - 1) / (m0 + n - j) * gamma[j]
            )
        return (mu1, gamma1)

    def f_hk(IC_ARL, k):
        a = np.zeros(9)  # coeff of polynomial
        if IC_ARL == 30:
            a = np.array(
                [
                    13.54844243,
                    -278.74909693,
                    953.99916442,
                    -1436.40494624,
                    1161.01248598,
                    -535.16425988,
                    141.95524256,
                    -24.57009576,
                    5.02083018,
                ]
            )
            h_k = 0
        for i in range(a.shape[0]):
            h_k = h_k + a[i] * k ** (a.shape[0] - 1 - i)
        return h_k

    def f_Yt(Rst_t, Rst_tminus1):
        Y_t = (Rst_t + Rst_tminus1) / 2
        return Y_t

    def f_Rst(x):
        t = x.size
        x_t = x[-1]
        R_t = np.sum(x_t >= x)
        E = (t + 1) / 2
        V = ((t + 1) * (t - 1)) / 12
        Rst_t = (R_t - E) / np.sqrt(V)
        return Rst_t

    X_df = df.to_numpy()
    X_star = np.zeros(X_df.shape[0])
    Rst = np.zeros(X_df.shape[0])
    S = np.zeros(df.shape)

    X_star[0:m], mu, gamma, dw_stat = f_initial_decoorelation(X_df[0:m], m, bmax)
    # print(dw_stat)
    Tau = 0
    for i in range(m, X_df.shape[0]):
        Xstar = X_star[0:i]

        Xt = X_df[i][0]
        Xt_star = f_Xtstar(Xstar, Xt, mu, gamma, Tau, i, m)
        X_star[i] = Xt_star

        Rst[i] = f_Rst(X_star[0 : i + 1])
        Yt = f_Yt(Rst[i], Rst[i - 1])
        delta_t = max(Yt, delta_0)
        k = delta_t / 2
        hk = f_hk(IC_ARL, k)
        z = S[i - 1, :] + ((Rst[i] - k) / hk)
        S[i, :] = max(0, z)

        if S[i, :] == 0:
            Tau = 0
        else:
            Tau = min(Tau + 1, bmax)

        mu, gamma = f_moments(X_star, Xt_star, mu, gamma, i, m)

    S = np.reshape(S, df.shape)
    df_S = pd.DataFrame(S, columns=df.columns, index=df.index)
    leak_det = pd.Series(dtype=object)

    for i, pipe in enumerate(df_S):
        hthr = h_thr_alter
        if any(df_S[pipe] > hthr):
            leak_det[pipe] = df_S.index[(df_S[pipe] > hthr).values][0]
            det = 1
            leak_det = leak_det[0]
        else:
            det = 0

    return (leak_det, det, S)


# %%
