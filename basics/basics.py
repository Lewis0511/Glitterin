"""
basics.py
Author: Ziyang Liu @ Glitterin.
Second Commit. 2024.03.22.
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
plt.style.use('ggplot')
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


def basics():
    """
    TODO.

    :return: None
    :raise: None
    """
    titles = ['MAE', 'MSE', 'RMSE', '$R$-Squared', '$r$ coefficient']
    filenames = ['面粉含水率']  # , '苹果甜度']
    data = defaultdict()
    spa_norms = defaultdict()
    uve_coefs = defaultdict()

    for filename in filenames:

        df = pd.read_excel('data/%s.xlsx' % filename)

        X = df.iloc[:, : -1].to_numpy().astype(np.float64)
        Y = df.iloc[:,   -1].to_numpy().astype(np.float64)

        X = savgol_filter(X, 15, 5)
        X = StandardScaler().fit_transform(X.T).T
        # x_ave = np.average(X, 0)
        # x_als = utils.ALS(x_ave, 0.01, 1, 3)
        # X = np.array([x - utils.ALS(x, 0.01, 1, 3) for x in X])

        plt.figure(figsize=(24, 8))
        plt.plot(WL_1899, X.T)
        plt.xticks(wl_1899)
        plt.axvline( 760, alpha=0.5, color='black', linestyle='--')
        plt.axvline( 847, alpha=0.2, color='black', linestyle='--')  # [ 845,  850]
        plt.axvline( 970, alpha=0.5, color='black', linestyle='--')
        plt.axvline(1190, alpha=0.5, color='black', linestyle='--')
        plt.axvline(1425, alpha=0.5, color='black', linestyle='--')  # [1400, 1450]
        plt.axvline(1925, alpha=0.5, color='black', linestyle='--')  # [1900, 1950]
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Absorbance')
        plt.title('%s Spectra' % filename)
        plt.savefig('figure/%s' % filename + '_spectra')
        plt.close()

        train_size = 64 if filename == '面粉含水率' else 512 if filename == '苹果甜度' else None
        X_train, X_test = X[:train_size, :], X[train_size:, :]
        Y_train, Y_test = Y[:train_size]   , Y[train_size:]

        data[filename] = [X_train, X_test, Y_train, Y_test]

        # spa_norms[name] = utils.SPA(X)
        # uve_coefs[name] = utils.UVE(X, Y)

    for name, norms in spa_norms.items():

        plt.figure(figsize=(24, 8))
        plt.plot(WL_0884, norms)
        plt.xticks(wl_0884)
        plt.yscale('log')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Accumulated SPA Norms')
        plt.title('%s SPA Norms' % name, fontproperties=prop)
        plt.savefig('figure/%s' % name + '_spa_norms')
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
        plt.title('%s UVE Coefficients' % name, fontproperties=prop)
        plt.savefig('figure/%s' % name + '_uve_coefs')
        plt.close()

    metrics = defaultdict(list)
    components = np.concatenate([np.arange(3, 17), np.arange(1, 3)])

    for name, [X_train, X_test, Y_train, Y_test] in data.items():

        X_mean = np.mean(X_train, 0)
        X_train -= X_mean
        X_test  -= X_mean

        pls_model, pls_n_components = utils.PLS(X_train, X_test, Y_train, False, False)
        pcr_model, pcr_n_components = utils.PCR(X_train, X_test, Y_train, False, False)

        pca = PCA(n_components=pcr_n_components)
        pca.fit(X_train)

        Y_train_pred_pls = pls_model.predict(X_train).squeeze()
        Y_test_pred_pls  = pls_model.predict(X_test) .squeeze()

        Y_train_pred_pcr = pcr_model.predict(pca.transform(X_train)).squeeze()
        Y_test_pred_pcr  = pcr_model.predict(pca.transform(X_test)) .squeeze()

        utils.plot_prediction_pls_and_pcr \
            (Y_train, Y_train_pred_pls, Y_train_pred_pcr, Y_test, Y_test_pred_pls, Y_test_pred_pcr, None, None,
             pls_n_components, pcr_n_components, name, '%s' % name + '_pls_pcr_predictions', False)

        for n_components in components:

            pls_model = PLSRegression(n_components, scale=False)
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
                Y_test, Y_test_pred_pls, Y_test_pred_pcr, None, None, n_components, n_components))

        Y_test = np.expand_dims(Y_test, -1)
        pls_model_x_scores_ = X_test @ pls_model.x_loadings_
        pls_model_y_scores_ = Y_test @ pls_model.y_loadings_

        plt.figure(figsize=(24, 8))
        plt.plot(WL_1899, pls_model.x_loadings_[:, 0], label='$\mathbf{p}_1$')
        plt.plot(WL_1899, pls_model.x_loadings_[:, 1], label='$\mathbf{p}_2$')
        plt.xticks(wl_1899)
        plt.axvline( 760, alpha=0.5, color='black', linestyle='--')
        plt.axvline( 847, alpha=0.2, color='black', linestyle='--')  # [ 845,  850]
        plt.axvline( 970, alpha=0.5, color='black', linestyle='--')
        plt.axvline(1190, alpha=0.5, color='black', linestyle='--')
        plt.axvline(1425, alpha=0.5, color='black', linestyle='--')  # [1400, 1450]
        plt.axvline(1925, alpha=0.5, color='black', linestyle='--')  # [1900, 1950]
        plt.legend()
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Loadings')
        plt.title('%s Fig 1. The PLS loadings $\mathbf{p}_1$ and $\mathbf{p}_2$, training set.' % name)
        plt.savefig('figure/%s' % name + '_fig_01')
        plt.close()

        min_score = np.min(np.concatenate([pls_model.x_scores_[:, 0], pls_model.y_scores_[:, 0]]))
        max_score = np.max(np.concatenate([pls_model.x_scores_[:, 0], pls_model.y_scores_[:, 0]]))
        ideal = np.linspace(min_score, max_score, 2)

        plt.figure(figsize=(8, 8))
        plt.plot(ideal, ideal, alpha=0.5, color='black')
        plt.scatter(pls_model.x_scores_[:, 0], pls_model.y_scores_[:, 0], 4, 1 - Y_train, cmap='seismic')
        plt.axhline(alpha=0.5, color='k')
        plt.axvline(alpha=0.5, color='k')
        plt.xlabel('$\mathbf{t}_1$')
        plt.ylabel('$\mathbf{u}_1$')
        plt.title('%s Fig 2. The PLS scores $\mathbf{t}_1$ and $\mathbf{u}_1$, training set.' % name)
        plt.savefig('figure/%s' % name + '_fig_02')
        plt.close()

        min_score = np.min(np.concatenate([pls_model_x_scores_[:, 0], pls_model_y_scores_[:, 0]]))
        max_score = np.max(np.concatenate([pls_model_x_scores_[:, 0], pls_model_y_scores_[:, 0]]))
        np.linspace(min_score, max_score, 2)

        plt.figure(figsize=(8, 8))
        plt.plot(ideal, ideal, alpha=0.5, color='black')
        plt.scatter(pls_model_x_scores_[:, 0], pls_model_y_scores_[:, 0], 4, 1 - Y_test, cmap='seismic')
        plt.axhline(alpha=0.5, color='k')
        plt.axvline(alpha=0.5, color='k')
        plt.xlabel('$\mathbf{t}_1$')
        plt.ylabel('$\mathbf{u}_1$')
        plt.title('%s Fig 3. The PLS scores $\mathbf{t}_1$ and $\mathbf{u}_1$, testing set.' % name)
        plt.savefig('figure/%s' % name + '_fig_03')
        plt.close()

        plt.figure(figsize=(8, 8))
        plt.scatter(pls_model.x_scores_[:, 0], pls_model.x_scores_[:, 1], 4, 1 - Y_train, cmap='seismic')
        plt.axhline(alpha=0.5, color='k')
        plt.axvline(alpha=0.5, color='k')
        plt.xlabel('$\mathbf{t}_1$')
        plt.ylabel('$\mathbf{t}_2$')
        plt.title('%s Fig 4. The PLS scores $\mathbf{t}_1$ and $\mathbf{t}_2$, training set.' % name)
        plt.savefig('figure/%s' % name + '_fig_04')
        plt.close()

        plt.figure(figsize=(8, 8))
        plt.scatter(pls_model_x_scores_[:, 0], pls_model_x_scores_[:, 1], 4, 1 - Y_test, cmap='seismic')
        plt.axhline(alpha=0.5, color='k')
        plt.axvline(alpha=0.5, color='k')
        plt.xlabel('$\mathbf{t}_1$')
        plt.ylabel('$\mathbf{t}_2$')
        plt.title('%s Fig 5. The PLS scores $\mathbf{t}_1$ and $\mathbf{t}_2$, testing set.' % name)
        plt.savefig('figure/%s' % name + '_fig_05')
        plt.close()

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

        plt.savefig('figure/%s' % name + '_pls_pcr_n_components')
        plt.close()

    return


basics()