"""
iPPG.py
Author: Ziyang Liu @ Glitterin.
Initial Commit. 2024.03.15.
《基于深度学习的多光谱容积脉搏波的血压测量技术研究》.
孙笑敩，中国科学院西安光学精密机械研究所，博士学位论文.
"""
import PyEMD
import utils

import numpy  as np
import pandas as pd
import scipy  as sp

from matplotlib   import font_manager
from matplotlib   import pyplot as plt
from matplotlib   import ticker
from scipy.signal import savgol_filter

prop = font_manager.FontProperties()
prop.set_size(16)
prop.set_family('SimHei')


def iPPG():
    """
    TODO.

    :return: Y, Y_pred.
    :raise: None.
    """
    data_0 = pd.read_excel('data/脉动时域光谱响应值.xlsx', 0)
    data_1 = pd.read_excel('data/脉动时域光谱响应值.xlsx', 1)

    T = data_0.iloc[:, 0 ].astype(np.float64).to_numpy()
    X = data_0.iloc[:, 1:].astype(np.float64).to_numpy().T
    Y = data_1.iloc[:, 1 ].astype(np.float64).to_numpy()

    (N, K), M = X.shape, 1

    Y_pred = [None] * N

    for i, x in enumerate(X):

        x_eemd    = EEMD(x, T, True, False)
        x_fft     = sp.fft.fft(x_eemd)[: K // 2]
        Y_pred[i] = np.argmax(np.abs(x_fft)) / T[-1] * 60

        # plt.figure(figsize=(8, 8))
        # plt.plot([i / T[-1] for i in range(K // 2)], x_fft)

    print(Y)
    print(Y_pred)

    utils.plot_prediction(Y, Y_pred, Y, Y_pred, None, None, 'iPPG Heart Rate', 'iPPG_heart_rate.png', False)

    return Y, Y_pred


def EEMD(s, t, save=None, plot=False):
    """
    多光谱 iPPG 信号获取与去噪算法.
    Multi-spectral iPPG signal acquisition and denoising algorithm.

    :param s: 原始 iPPG 信号.
    :param t: 时间（秒）.
    :param save: Filename to save. Default None.
    :param plot: Plot if True. Default False.
    :return: 优化后的光谱 iPPG 时序信号.
    :raise: None.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(t, s, color='blue')
    plt.xticks(range(0, 24, 4))
    plt.xlabel('Time / s')
    plt.title('图 4-10 iPPG原始信号图.\nFigure 4-10 iPPG Raw Signal.', fontproperties=prop)

    if save: plt.savefig('figure/%s' % 'iPPG_raw_signal')
    if plot: plt.show()
    if True: plt.close()

    K = len(s)
    # d = []  # 微分信号
    # D = []  # 积分信号
    # T = []  # 平滑滤波后信号
    # E = []  # 去除基线漂移信号
    # P = []  # 重构后信号

    D2 = sp.sparse.diags([(i % 2 * 2 - 1) * sp.special.binom(2 - 1, i) for i in range(2)], [-i for i in range(2)], (K, K - 2 + 1))

    # d_orig = D2.T.dot(s)
    # d_mean = np.mean(d_orig)
    # d_std  = np.std (d_orig)

    # bland_altman_plot(d_orig)

    """
    Conclusion from Bland-Altman Analysis is that
    no mutation noise points should be detected.
    """
    d = D2.T.dot(s)

    # z = sp.stats.norm.ppf(1 - 2 ** -2 / 10)
    # d = [d_orig[i] if i in [0, K - 2] or abs(d_orig[i] - d_mean) <= z * d_std
    #      else (d_orig[i - 1] + d_orig[i + 1]) / 2 for i in range(K - 1)]

    D = np.cumsum(d)
    T = savgol_filter(D, 15, 5)

    """
    集合经验模态分解.
    Ensemble Empirical Mode Decomposition (EEMD).
    https://pyemd.readthedocs.io/en/latest/examples.html.
    """
    eemd = PyEMD.EEMD()

    eIMFs = eemd.eemd(T)
    nIMFs = eIMFs.shape[0]

    P = eIMFs[1] + eIMFs[2]

    fig, axs = plt.subplots(nIMFs+2, 1, figsize=(8, 8))

    for ax in axs: ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%+g'))

    axs[0].plot(T)
    axs[0].locator_params(axis='y', nbins=2)
    axs[0].set_title('去除高频噪声后的信号', fontproperties=prop, fontsize=8)

    for n in range(nIMFs):
        axs[n + 1].plot(eIMFs[n])
        axs[n + 1].set_ylabel('IMF %i' % (n + 1))
        axs[n + 1].locator_params(axis='y', nbins=2)

    axs[-1].plot(T - np.sum(eIMFs, axis=0))
    axs[-1].set_ylabel('Res')
    axs[-1].locator_params(axis='y', nbins=2)

    plt.tight_layout()

    if save: plt.savefig('figure/%s' % 'EEMD_IMFs')
    if plot: plt.show()
    if True: plt.close()

    return P


def bland_altman_plot(x, save=None, plot=False):
    """
    Bland-Altman Plot.
    https://en.wikipedia.org/wiki/Bland%E2%80%93Altman_plot.
    https://stackoverflow.com/questions/16399279/bland-altman-plot-in-python.

    :param x: TODO.
    :param save: Filename to save. Default None.
    :param plot: Plot if True. Default False.
    :return: None.
    :raise: None.
    """
    x_mean = np.mean(x)
    x_std  = np.std (x)
    z_neg = sp.stats.norm.ppf(1 - 2 ** -8 / 10)  # REALLY ???
    z_pos = sp.stats.norm.ppf(1 - 2 ** -2 / 10)

    plt.plot(x)
    plt.axhline(x_mean, color='tab:gray', linestyle='--')
    plt.axhline(x_mean - z_neg * x_std, color='tab:gray', linestyle='--')
    plt.axhline(x_mean + z_pos * x_std, color='tab:gray', linestyle='--')

    plt.title('Bland-Altman Plot')

    if save: plt.savefig('figure/%s' % save)
    if plot: plt.show()
    if True: plt.close()

    return


iPPG()