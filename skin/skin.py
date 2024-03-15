"""
skin.py
Author: Ziyang Liu @ Glitterin.
Initial Commit. 2024.03.15.
"""
import utils

import numpy  as np
import pandas as pd
import scipy  as sp

from collections  import defaultdict
from matplotlib   import font_manager
from matplotlib   import pyplot as plt
from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition       import PCA
from sklearn.linear_model        import LinearRegression
from sklearn.preprocessing       import StandardScaler

prop = font_manager.FontProperties()
prop.set_size(16)
prop.set_family('SimHei')

"""https://stackoverflow.com/questions/65493638/glyph-23130-missing-from-current-font."""
plt.rcParams['font.sans-serif'] = ['SimHei']  # show Chinese label
plt.rcParams['axes.unicode_minus'] = False    # these two lines need to be set manually

"""
IN CASE YOU ARE CONFUSED

WL_0884: [1240, 1700], len =  884
WL_1768: [1700, 2160], len =  884
WL_1899: [ 866, 2530], len = 1899

wl_0884: [1240, 1700], len = 19, labels for ticks
wl_1768: [1240, 2160], len = 20, labels for ticks
wl_0884: [ 866, 2530], len = 20, labels for ticks
"""

WL_0884 = pd.read_excel('../wl_0884.xlsx', 0, header=None).iloc[:, :].to_numpy().astype(np.float64)  # [1240, 1700] nm
WL_1899 = pd.read_excel('../wl_1899.xlsx', 1, header=None).iloc[:, 3].to_numpy().astype(np.float64)  # [ 866, 2530] nm
WL_0884 = WL_0884.squeeze()
WL_1768 = [WL - WL_0884[0] + WL_0884[-1] for WL in WL_0884]

wl_0884 = np.concatenate([np.arange(0,  883,  50),  [883]])
wl_1899 = np.concatenate([np.arange(0, 1899, 100), [1898]])

wl_0884 = [int(wl) for wl in WL_0884[wl_0884]]
wl_1899 = [int(wl) for wl in WL_1899[wl_1899]]
wl_1768 = np.concatenate([np.array(wl_0884[::2]), np.array([wl - wl_0884[0] + wl_0884[-1] for wl in wl_0884[::2]])])


def skin():
    """
    TODO.

    :return: None
    :raise: None
    """
    titles = ['MAE', 'MSE', 'RMSE', '$R$-Squared', '$r$ coefficient']
    filenames = ['刘0304.xlsx', '卓0304.xlsx', '吴0304.xlsx', '温0304.xlsx', '王0304.xlsx']
    data = {filename[:5]: [pd.read_excel(filename, i, header=None) for i in range(3)] for filename in filenames}
    data['卓0304'][2].iloc[257, 2] /= 10  # typo correction
    slices = {'刘0304': [[0, 100], [100, 160], [160, 250], [250, 300]],
              '卓0304': [[0, 101], [101, 151], [151, 241], [241, 291]],
              '吴0304': [[0, 110], [110, 170], [170, 260], [260, 310]],
              '温0304': [[0, 110], [110, 170], [170, 260], [260, 310]],
              '王0304': [[0, 110], [110, 170], [170, 260], [260, 310]]}

    spa_norms = defaultdict()
    uve_coefs = defaultdict()

    for name, [T, X, Y] in data.items():
        # TODO: (T, Y[0]), (T, Y[1]), (X, Y[0]), (X, Y[1])
        T = T.to_numpy().astype(np.float64).T
        X = X.to_numpy().astype(np.float64).T
        Y = Y.iloc[1:, 2].to_numpy().astype(np.float64)

        T = StandardScaler().fit_transform(T.T).T
        X = StandardScaler().fit_transform(X.T).T

        for [l, r] in slices[name]:

            t_ave = np.average(T[l:r, :], 0)
            x_ave = np.average(X[l:r, :], 0)

            t_als = utils.ALS(t_ave, 0.01, 1, 3)
            x_als = utils.ALS(x_ave, 0.01, 1, 3)

            T[l:r, :] -= t_als
            X[l:r, :] -= x_als

            # plt.figure(figsize=(24, 8))
            # plt.plot(t_ave, label='t_ave')
            # plt.plot(t_als, label='t_als')
            # plt.legend()
            # plt.title('%s T[%i : %i]' % (name, l, r), fontproperties=prop)

            # plt.figure(figsize=(24, 8))
            # plt.plot(x_ave, label='x_ave')
            # plt.plot(x_als, label='x_als')
            # plt.legend()
            # plt.title('%s X[%i : %i]' % (name, l, r), fontproperties=prop)

        X = savgol_filter(X, 15, 5)

        T_train, T_test = T[:-50, :], T[-50:, :]
        X_train, X_test = X[:-50, :], X[-50:, :]
        Y_train, Y_test = Y[:-50], Y[-50:]

        data[name] = [T_train, T_test, X_train, X_test, Y_train, Y_test]

        # spa_norms[name] = utils.SPA(X)
        # uve_coefs[name] = utils.UVE(X, Y)

    for name, norms in spa_norms.items():

        plt.figure(figsize=(24, 8))
        plt.plot(WL_0884, norms)
        plt.xticks(wl_0884)
        plt.yscale('log')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Accumulated SPA Norms')
        plt.title('%s Recon SPA Norms' % name, fontproperties=prop)
        plt.savefig('%s_recon_spa_norms.png' % name)
        plt.close()

    for name, coefs in uve_coefs.items():

        plt.figure(figsize=(24, 8))
        plt.plot(WL_0884, coefs[:884], color='tab:blue', label='Measured')
        plt.plot(WL_1768, coefs[884:], color='tab:red' , label='Random')
        plt.xticks(wl_1768, np.concatenate([wl_0884[::2], wl_0884[::2]]))
        plt.axvline(WL_0884[-1], color='black', linestyle='--')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('UVE Coefficients / Stability')
        plt.legend()
        plt.title('%s Recon vs. Moisture UVE Coefficients' % name, fontproperties=prop)
        plt.savefig('%s_recon_moisture_uve_coefs.png' % name)
        plt.close()

    metrics = defaultdict(list)
    components = np.arange(1, 17)

    for name, [_, _, X_train, X_test, Y_train, Y_test] in data.items():

        pls_model, pls_n_components = utils.PLS(X_train, X_test, Y_train, False, False)
        pcr_model, pcr_n_components = utils.PCR(X_train, X_test, Y_train, False, False)

        pca = PCA(n_components=pcr_n_components)
        pca.fit(X_train)

        Y_train_pred_pls = pls_model.predict(X_train).squeeze()
        Y_test_pred_pls  = pls_model.predict(X_test) .squeeze()

        Y_train_pred_pcr = pcr_model.predict(pca.transform(X_train)).squeeze()
        Y_test_pred_pcr  = pcr_model.predict(pca.transform(X_test)) .squeeze()

        utils.plot_prediction_pls_and_pcr \
            (Y_train, Y_train_pred_pls, Y_train_pred_pcr, Y_test, Y_test_pred_pls, Y_test_pred_pcr, 0.0, 0.6,
             pls_n_components, pcr_n_components, name, '%s_recon_moisture_pls_pcr_predictions.png' % name, False)

        for n_components in components:

            pls_model = PLSRegression(n_components)
            pls_model.fit(X_train, Y_train)

            Y_train_pred_pls = pls_model.predict(X_train).squeeze()
            Y_test_pred_pls  = pls_model.predict(X_test) .squeeze()

            pca = PCA(n_components)
            pca.fit(X_train)

            T_train = pca.transform(X_train)
            T_test  = pca.transform(X_test)

            pcr_model = LinearRegression()
            pcr_model.fit(T_train, Y_train)

            Y_train_pred_pcr = pcr_model.predict(T_train).squeeze()
            Y_test_pred_pcr  = pcr_model.predict(T_test) .squeeze()

            metrics[name].append(utils.plot_prediction_pls_and_pcr(Y_train, Y_train_pred_pls, Y_train_pred_pcr,
                Y_test, Y_test_pred_pls, Y_test_pred_pcr, 0.0, 0.6, n_components, n_components))

    for name, metric in metrics.items():

        figure, axes = plt.subplots(5, 2, figsize=(32, 40), squeeze=False)

        axes[0, 0].set_title('%s PLS Models' % name, fontproperties=prop)
        axes[0, 1].set_title('%s PCR Models' % name, fontproperties=prop)

        for row in range(5):

            axes[row, 0].plot(components, [metric[i][row * 2 + 0] for i in range(16)], label='training')
            axes[row, 0].plot(components, [metric[i][row * 2 + 1] for i in range(16)], label='testing')
            axes[row, 0].set_xlabel('n_components')
            axes[row, 0].set_ylabel(titles[row])

            axes[row, 1].plot(components, [metric[i][row * 2 + 10] for i in range(16)], label='training')
            axes[row, 1].plot(components, [metric[i][row * 2 + 11] for i in range(16)], label='testing')
            axes[row, 1].set_xlabel('n_components')
            axes[row, 1].set_ylabel(titles[row])

            axes[row, 0].legend()
            axes[row, 1].legend()

        plt.savefig('%s_recon_moisture_pls_pcr_n_components.png' % name)
        plt.close()

    return


skin()