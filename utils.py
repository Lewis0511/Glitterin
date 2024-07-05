"""
utils.py
Author: Ziyang Liu @ Glitterin
Updated 2024.07.05
"""

import datetime
import PyEMD

import numpy  as np
import pandas as pd
import scipy  as sp

from matplotlib   import font_manager
from matplotlib   import pyplot as plt
from matplotlib   import ticker
from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition       import PCA
from sklearn.linear_model        import LinearRegression
from sklearn.metrics             import mean_absolute_error
from sklearn.metrics             import mean_squared_error
from sklearn.metrics             import r2_score
from sklearn.metrics             import root_mean_squared_error
from sklearn.model_selection     import cross_val_predict

prop = font_manager.FontProperties()
prop.set_size(16)
prop.set_family(['DejaVu Sans', 'SimHei'])

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

WL_0884 = pd.read_excel('wl_0884.xlsx', 0, header=None).iloc[:, :].to_numpy().astype(np.float64)  # [1240, 1700] nm
WL_1899 = pd.read_excel('wl_1899.xlsx', 1, header=None).iloc[:, 3].to_numpy().astype(np.float64)  # [ 866, 2530] nm
WL_0884 = WL_0884.squeeze()
WL_1768 = [WL - WL_0884[0] + WL_0884[-1] for WL in WL_0884]

wl_0884 = np.concatenate([np.arange(0,  883,  50),  [883]])
wl_1899 = np.concatenate([np.arange(0, 1899, 100), [1898]])

wl_0884 = [int(wl) for wl in WL_0884[wl_0884]]
wl_1899 = [int(wl) for wl in WL_1899[wl_1899]]
wl_1768 = np.concatenate([np.array(wl_0884[::2]), np.array([wl - wl_0884[0] + wl_0884[-1] for wl in wl_0884[::2]])])


def curr_date(): return datetime.datetime.now().strftime('%Y%m%d')
def curr_time(): return datetime.datetime.now().strftime('%Y/%m/%d %H:%M')


def mean_centering(X, axis=None):
    """
    Mean-center the matrix X.

    :param X: N x K matrix where (N, K) == (num_samples, num_features).
    :param axis: (None, 0, 1) == (element, column, row)-wise mean-centering. Default None.
    :return: Your new X.
    :raise: ValueError: if axis is not None, 0, or 1.
    """
    if axis not in [None, 0, 1]: raise ValueError('axis should be None, 0, or 1.')

    X_mean = np.mean(X, axis)

    if axis is not None: X_mean = np.expand_dims(X_mean, axis)

    return X - X_mean


def normalization(X, axis=None):
    """
    Normalize the matrix X.

    :param X: N x K matrix where (N, K) == (num_samples, num_features).
    :param axis: (None, 0, 1) == (element, column, row)-wise normalization. Default None.
    :return: Your new X.
    :raise: ValueError: if axis is not None, 0, or 1.
    """
    if axis not in [None, 0, 1]: raise ValueError('axis should be None, 0, or 1.')

    X_min = np.min(X, axis)
    X_max = np.max(X, axis)

    if axis is not None:
        X_min = np.expand_dims(X_min, axis)
        X_max = np.expand_dims(X_max, axis)

    return (X - X_min) / (X_max - X_min)


def standardization(X, axis=None, mean_center=True):
    """
    Standardize the matrix X.
    X_snv == standardization(X, axis=1, mean_center=True) == StandardScaler().fit_transform(X.T).T

    :param X: N x K matrix where (N, K) == (num_samples, num_features).
    :param axis: (None, 0, 1) == (element, column, row)-wise standardization. Default None.
    :param mean_center: Mean-center the matrix X if True. Default True.
    :return: Your new X.
    :raise: ValueError: if axis is not None, 0, or 1.
    """
    if axis not in [None, 0, 1]: raise ValueError('axis should be None, 0, or 1.')

    X_mean = np.mean(X, axis)
    X_std  = np.std (X, axis)

    if axis is not None:
        X_mean = np.expand_dims(X_mean, axis)
        X_std  = np.expand_dims(X_std , axis)

    return (X - X_mean) / X_std if mean_center else X / X_std


def MSC(X, mean_center=True, reference=None):
    """
    Multiplicative Scatter Correction.

    :param X: N x K matrix where (N, K) == (num_samples, num_features).
    :param mean_center: Row-wise mean-centering the matrix X if True. Default True.
    :param reference: Reference spectrum. Default None.
    :return: Your new X.
    :raise: None.
    """
    if mean_center: X = mean_centering(X, 1)

    X_msc = np.empty(X.shape)
    X_ref = np.mean(X, 0) if reference is None else reference

    for i in range(X.shape[0]):
        a, b = np.polyfit(X_ref, X[i], 1)
        X_msc[i] = (X[i] - b) / a

    return X_msc


def ALS(x, p, lam, d, n_iter=10):
    """
    Asymmetric Least Squares.
    https://pubs.acs.org/doi/10.1021/ac034173t.
    https://pubs.acs.org/doi/10.1021/ac034800e.
    https://www.sciencedirect.com/science/article/pii/S0003267010010627.

    :param x: Spectrum.
    :param p: Asymmetric penalty weight: wi = p if xi > zi and wi = 1 - p otherwise.
    :param lam: Roughness penalty parameter. The larger lambda, the smoother the estimated baseline.
    :param d: Order of the difference matrix in the roughness penalty.
    :param n_iter: Number of iterations. Default 10.
    :return: Estimated baseline of the spectrum.
    :raise: ValueError: if n_iter is not a positive integer.
    """
    if n_iter <= 0: raise ValueError('n_iter should be a positive integer.')
    L = len(x)
    D = sp.sparse.diags([(1 - i % 2 * 2) * sp.special.binom(d - 1, i) for i in range(d)], [-i for i in range(d)], (L, L - d + 1))
    D = lam * D.dot(D.T)
    w = np.ones(L)
    z = None

    for _ in range(n_iter):

        W = sp.sparse.diags(w)
        Z = W + lam * D.dot(D.T)
        z = sp.sparse.linalg.spsolve(Z, w * x)
        w = p * (x > z) + (1 - p) * (x < z)

    return z


def plot_prediction(Y_train, Y_train_pred, Y_test, Y_test_pred, Y_min, Y_max, title='', save=None, plot=False):
    """
    Plot prediction.

    :param Y_train: TODO.
    :param Y_train_pred: TODO.
    :param Y_test: TODO.
    :param Y_test_pred: TODO.
    :param Y_min: TODO.
    :param Y_max: TODO.
    :param title: Title of the plot. Default an empty string.
    :param save: Filename to save. Default None.
    :param plot: Plot if True. Default False.
    :return: mae_train, mae_test, mse_train, mse_test, rmse_train, rmse_test, r2_train, r2_test, r_train, r_test.
    :raise: None.
    """
    mae_train  = mean_absolute_error(Y_train, Y_train_pred)
    mae_test   = mean_absolute_error(Y_test , Y_test_pred)

    mse_train  = mean_squared_error(Y_train, Y_train_pred)
    mse_test   = mean_squared_error(Y_test , Y_test_pred)

    rmse_train = root_mean_squared_error(Y_train, Y_train_pred)
    rmse_test  = root_mean_squared_error(Y_test , Y_test_pred)

    r2_train   = r2_score(Y_train, Y_train_pred)
    r2_test    = r2_score(Y_test , Y_test_pred)

    r_train    = np.corrcoef(Y_train, Y_train_pred)[0, 1]
    r_test     = np.corrcoef(Y_test , Y_test_pred) [0, 1]

    if Y_min is None: Y_min = np.min(np.concatenate([Y_train, Y_train_pred, Y_test, Y_test_pred]))
    if Y_max is None: Y_max = np.max(np.concatenate([Y_train, Y_train_pred, Y_test, Y_test_pred]))

    plt.figure(figsize=(8, 8))
    plt.xlim(Y_min, Y_max)
    plt.ylim(Y_min, Y_max)

    ideal = np.linspace(Y_min, Y_max, 2)
    fitted = np.polyval(np.polyfit(Y_test, Y_test_pred, 1), ideal)

    plt.scatter(Y_train, Y_train_pred, 8, 'pink')
    plt.scatter(Y_test , Y_test_pred , 8, 'red')

    plt.plot(ideal, ideal , 'g', label='ideal')
    plt.plot(ideal, fitted, 'b', label='fitted')

    plt.text(0.10, 0.08, 'Training: MAE = %.4f' % mae_train + ', RMSE = %.4f' % rmse_train + ', $R^2$ = %.4f' % r2_train + ', $r_c$ = %.4f' % np.float64(r_train), transform=plt.gca().transAxes)
    plt.text(0.10, 0.04, ' Testing: MAE = %.4f' % mae_test  + ', RMSE = %.4f' % rmse_test  + ', $Q^2$ = %.4f' % r2_test  + ', $r_p$ = %.4f' % np.float64(r_test ), transform=plt.gca().transAxes)

    plt.xlabel( 'Measured Value')
    plt.ylabel('Predicted Value')

    plt.legend(loc='upper left')
    plt.title(title, fontproperties=prop)

    if save: plt.savefig(save)
    if plot: plt.show()
    if True: plt.close()

    return mae_train, mae_test, mse_train, mse_test, rmse_train, rmse_test, r2_train, r2_test, r_train, r_test


def plot_prediction_pls_and_pcr(Y_train, Y_train_pred_pls, Y_train_pred_pcr, Y_test, Y_test_pred_pls, Y_test_pred_pcr,
                                Y_min, Y_max, title='', save=None, plot=False):
    """
    Plot PLS and PCR Predictions.

    :param Y_train: TODO.
    :param Y_train_pred_pls: TODO.
    :param Y_train_pred_pcr: TODO.
    :param Y_test: TODO.
    :param Y_test_pred_pls: TODO.
    :param Y_test_pred_pcr: TODO.
    :param Y_min: TODO.
    :param Y_max: TODO.
    :param title: Title of the plots. Default an empty string.
    :param save: Filename to save. Default None.
    :param plot: Plot if True. Default False.
    :return: mae_train_pls, mae_test_pls, mse_train_pls, mse_test_pls, rmse_train_pls, rmse_test_pls, r2_train_pls, r2_test_pls, r_train_pls, r_test_pls,
             mae_train_pcr, mae_test_pcr, mse_train_pcr, mse_test_pcr, rmse_train_pcr, rmse_test_pcr, r2_train_pcr, r2_test_pcr, r_train_pcr, r_test_pcr.
    :raise: None
    """
    mae_train_pls = mean_absolute_error(Y_train, Y_train_pred_pls)
    mae_train_pcr = mean_absolute_error(Y_train, Y_train_pred_pcr)

    mae_test_pls = mean_absolute_error(Y_test, Y_test_pred_pls)
    mae_test_pcr = mean_absolute_error(Y_test, Y_test_pred_pcr)

    mse_train_pls = mean_squared_error(Y_train, Y_train_pred_pls)
    mse_train_pcr = mean_squared_error(Y_train, Y_train_pred_pcr)

    mse_test_pls = mean_squared_error(Y_test, Y_test_pred_pls)
    mse_test_pcr = mean_squared_error(Y_test, Y_test_pred_pcr)

    rmse_train_pls = root_mean_squared_error(Y_train, Y_train_pred_pls)
    rmse_train_pcr = root_mean_squared_error(Y_train, Y_train_pred_pcr)

    rmse_test_pls = root_mean_squared_error(Y_test, Y_test_pred_pls)
    rmse_test_pcr = root_mean_squared_error(Y_test, Y_test_pred_pcr)

    r2_train_pls = r2_score(Y_train, Y_train_pred_pls)
    r2_train_pcr = r2_score(Y_train, Y_train_pred_pcr)

    r2_test_pls = r2_score(Y_test, Y_test_pred_pls)
    r2_test_pcr = r2_score(Y_test, Y_test_pred_pcr)

    r_train_pls = np.corrcoef(Y_train, Y_train_pred_pls)[0, 1]
    r_train_pcr = np.corrcoef(Y_train, Y_train_pred_pcr)[0, 1]

    r_test_pls = np.corrcoef(Y_test, Y_test_pred_pls)[0, 1]
    r_test_pcr = np.corrcoef(Y_test, Y_test_pred_pcr)[0, 1]

    if Y_min is None: Y_min = np.min(np.concatenate([Y_train, Y_train_pred_pls, Y_train_pred_pcr, Y_test, Y_test_pred_pls, Y_test_pred_pcr]))
    if Y_max is None: Y_max = np.max(np.concatenate([Y_train, Y_train_pred_pls, Y_train_pred_pcr, Y_test, Y_test_pred_pls, Y_test_pred_pcr]))

    figure, axes = plt.subplots(1, 2, figsize=(16, 8), squeeze=False)
    ideal = np.linspace(Y_min, Y_max, 2)
    fitted_pls = np.polyval(np.polyfit(Y_test, Y_test_pred_pls, 1), ideal)
    fitted_pcr = np.polyval(np.polyfit(Y_test, Y_test_pred_pcr, 1), ideal)

    axis = axes[0, 0]
    axis.set_xlim(Y_min, Y_max)
    axis.set_ylim(Y_min, Y_max)
    axis.scatter(Y_train, Y_train_pred_pls, 8, 'pink')
    axis.scatter(Y_test , Y_test_pred_pls , 8, 'red')

    axis.plot(ideal, ideal     , 'g', label='ideal')
    axis.plot(ideal, fitted_pls, 'b', label='fitted')

    axis.text(0.10, 0.08, 'Training: MAE = %.4f' % mae_train_pls + ', RMSE = %.4f' % rmse_train_pls + ', $R^2$ = %.4f' % r2_train_pls + ', $r_c$ = %.4f' % np.float64(r_train_pls), transform=axis.transAxes)
    axis.text(0.10, 0.04, ' Testing: MAE = %.4f' % mae_test_pls  + ', RMSE = %.4f' % rmse_test_pls  + ', $Q^2$ = %.4f' % r2_test_pls  + ', $r_p$ = %.4f' % np.float64(r_test_pls ), transform=axis.transAxes)

    axis.set_xlabel( 'Measured Value')
    axis.set_ylabel('Predicted Value')

    axis = axes[0, 1]
    axis.set_xlim(Y_min, Y_max)
    axis.set_ylim(Y_min, Y_max)
    axis.scatter(Y_train, Y_train_pred_pcr, 8, 'pink')
    axis.scatter(Y_test , Y_test_pred_pcr , 8, 'red')

    axis.plot(ideal, ideal     , 'g', label='ideal')
    axis.plot(ideal, fitted_pcr, 'b', label='fitted')

    axis.text(0.10, 0.08, 'Training: MAE = %.4f' % mae_train_pcr + ', RMSE = %.4f' % rmse_train_pcr + ', $R^2$ = %.4f' % r2_train_pcr + ', $r_c$ = %.4f' % np.float64(r_train_pcr), transform=axis.transAxes)
    axis.text(0.10, 0.04, ' Testing: MAE = %.4f' % mae_test_pcr  + ', RMSE = %.4f' % rmse_test_pcr  + ', $Q^2$ = %.4f' % r2_test_pcr  + ', $r_p$ = %.4f' % np.float64(r_test_pcr ), transform=axis.transAxes)

    axis.set_xlabel( 'Measured Value')
    axis.set_ylabel('Predicted Value')

    axes[0, 0].legend()
    axes[0, 1].legend()

    if title: plt.suptitle(title, fontproperties=prop)
    if save : plt.savefig(save)
    if plot : plt.show()
    if True : plt.close()

    return mae_train_pls, mae_test_pls, mse_train_pls, mse_test_pls, rmse_train_pls, rmse_test_pls, r2_train_pls, r2_test_pls, r_train_pls, r_test_pls, \
           mae_train_pcr, mae_test_pcr, mse_train_pcr, mse_test_pcr, rmse_train_pcr, rmse_test_pcr, r2_train_pcr, r2_test_pcr, r_train_pcr, r_test_pcr


def optimize_pls_components(X, Y, cv):
    """

    :param X: TODO.
    :param Y: TODO.
    :param cv: TODO.
    :return: TODO.
    :raise: None.
    """
    mae, mse, rmse, r2, r = [], [], [], [], []

    for n_components in range(1, 17):

        model = PLSRegression(n_components)
        Y_pred = cross_val_predict(model, X, Y, cv=cv).squeeze()

        mae.append(mean_absolute_error(Y, Y_pred))
        mse.append(mean_squared_error(Y, Y_pred))
        rmse.append(root_mean_squared_error(Y, Y_pred))
        r2.append(r2_score(Y, Y_pred))
        r.append(np.corrcoef(Y, Y_pred)[0, 1])

    return np.array(mae), np.array(mse), np.array(rmse), np.array(r2), np.array(r)


def optimize_pca_components(X, Y, cv):
    """

    :param X: TODO.
    :param Y: TODO.
    :param cv: TODO.
    :return: TODO.
    :raise: None.
    """
    mae, mse, rmse, r2, r, X = [], [], [], [], [], PCA().fit_transform(X)

    for n_components in range(1, 17):

        model = LinearRegression()
        model.fit(X[:, :n_components], Y)
        Y_pred = cross_val_predict(model, X[:, :n_components], Y, cv=cv).squeeze()

        mae.append(mean_absolute_error(Y, Y_pred))
        mse.append(mean_squared_error(Y, Y_pred))
        rmse.append(root_mean_squared_error(Y, Y_pred))
        r2.append(r2_score(Y, Y_pred))
        r.append(np.corrcoef(Y, Y_pred)[0, 1])

    return np.array(mae), np.array(mse), np.array(rmse), np.array(r2), np.array(r)


def PLS(X_train, X_test, Y_train, return_coef, return_pred):
    """

    :param X_train: TODO.
    :param X_test: TODO.
    :param Y_train: TODO.
    :param return_coef: Return PLS model coefficients if True.
    :param return_pred: Return PLS model predictions if True.
    :return: See above. Otherwise, return PLS model.
    :raise: None.
    """
    counter = [0] * 16
    for cv in range(2, 17):
        mae, mse, rmse, r2, r = optimize_pls_components(X_train, Y_train, cv)
        counter[np.argmax(r2)] += 1
    n_components = np.argmax(counter) + 1

    model = PLSRegression(n_components)
    model.fit(X_train, Y_train)

    Y_train_pred = model.predict(X_train).squeeze()
    Y_test_pred  = model.predict(X_test) .squeeze()

    return model.coef_ if return_coef else (Y_train_pred, Y_test_pred) if return_pred else (model, n_components)


def PCR(X_train, X_test, Y_train, return_coef, return_pred):
    """

    :param X_train: TODO.
    :param X_test: TODO.
    :param Y_train: TODO.
    :param return_coef: Return PLS model coefficients if True.
    :param return_pred: Return PLS model predictions if True.
    :return: See above. Otherwise, return PLS model.
    :raise: None.
    """
    counter = [0] * 16
    for cv in range(2, 17):
        mae, mse, rmse, r2, r = optimize_pca_components(X_train, Y_train, cv)
        counter[np.argmax(r2)] += 1
    n_components = np.argmax(counter) + 1

    pca = PCA(n_components)
    pca.fit(X_train)

    T_train = pca.transform(X_train)
    T_test  = pca.transform(X_test)

    model = LinearRegression()
    model.fit(T_train, Y_train)

    Y_train_pred = model.predict(T_train).squeeze()
    Y_test_pred  = model.predict(T_test) .squeeze()

    return model.coef_ if return_coef else (Y_train_pred, Y_test_pred) if return_pred else (model, n_components)


def SPA(X, mean_center=True, title='', save=None, plot=False):
    """
    Successive Projections Algorithm.
    https://www.sciencedirect.com/science/article/pii/S0169743901001198.
    https://www.sciencedirect.com/science/article/pii/S0165993612002750.

    :param X: N x K matrix where (N, K) == (num_samples, num_features).
    :param mean_center: Mean center each column of X if True. Default True.
    :param title: Title of the plot. Default an empty string.
    :param save: Filename to save. Default None.
    :param plot: Plot if True. Default False.

    :return: SPA norms: K x 1 numpy array where K == num_features.
    :raise: None.
    """
    N, K = X.shape

    if K not in [884, 1899]: raise ValueError('Unrecognized number of wavelengths.')

    norms = np.zeros(K)

    if mean_center: X = mean_centering(X, axis=0)

    for i_init in range(K):

        X_copy = np.array(X)
        X_orth = [X[:, i_init]]

        n_select = [np.linalg.norm(X_orth[0])]
        i_select = [i_init]
        i_remain = [i for i in range(K) if i != i_init]

        for _ in range(min(N - mean_center, K) - 1):

            n_max = -1
            i_max = -1

            X_trans = np.array(X_orth).T

            for j in i_remain:

                xi = X_trans[:, -1]
                xj = X_copy[:, j]

                proj = xj - xi * np.dot(xi, xj) / np.dot(xi, xi)
                norm = np.linalg.norm(proj)

                X_copy[:, j] = proj
                if norm > n_max: n_max, i_max = norm, j

            X_orth.append(X_copy[:, i_max])

            n_select.append(n_max)
            i_select.append(i_max)
            i_remain.remove(i_max)

        for i in range(min(N - mean_center, K) - 1): norms[i_select[i]] += n_select[i]

    plt.figure(figsize=(24, 8))

    if K ==  884: plt.plot(WL_0884, norms)
    if K == 1899: plt.plot(WL_1899, norms)

    if K ==  884: plt.xticks(wl_0884)
    if K == 1899: plt.xticks(wl_1899)

    plt.yscale('log')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Accumulated SPA Norms')
    plt.title('%s SPA Norms' % title, fontproperties=prop)

    if save: plt.savefig('figure/%s' % save)
    if plot: plt.show()
    if True: plt.close()

    return norms


def UVE(X, Y, standardize=True, title='', save=None, plot=False):
    """
    Uninformative Variable Elimination.
    https://pubs.acs.org/doi/full/10.1021/ac960321m.
    https://www.sciencedirect.com/science/article/pii/S0169743907001980.

    :param X: N x K matrix where (N, K) == (num_samples, num_features).
    :param Y: M x K matrix where (M, K) == (num_labels , num_features).
    :param standardize: Standardize each column of X if True; mean-center each column of X if False. Default True.
    :param title: Title of the plot. Default an empty string.
    :param save: Filename to save. Default None.
    :param plot: Plot if True. Default False.

    :return: UVE coefficients: 2K x N matrix where (N, K) == (num_samples, num_features).
    :raise: None.
    """
    N, K = X.shape

    B = []
    R = []
    X = standardization(X, 0) if standardize else mean_centering(X, 0)
    T = np.random.rand(N, K)

    for i in range(N):

        X_train, X_test = np.concatenate([X[:i], X[i+1:]]), np.expand_dims(X[i], 0)
        T_train, T_test = np.concatenate([T[:i], T[i+1:]]), np.expand_dims(T[i], 0)
        Y_train, Y_test = np.concatenate([Y[:i], Y[i+1:]]), np.expand_dims(Y[i], 0)

        b_coef = PLS(X_train, X_test, Y_train, True, False)
        r_coef = PLS(T_train, T_test, Y_train, True, False)

        B.append(b_coef.squeeze())
        R.append(r_coef.squeeze())

    coefs = np.concatenate([np.array(B), np.array(R)], 1)
    coefs_mean = np.mean(coefs, 0)
    coefs_std  = np.std (coefs, 0)
    coefs = coefs_mean / coefs_std

    if not plot: return coefs

    plt.figure(figsize=(24, 8))

    if K ==  884: plt.plot(WL_0884, coefs[:884], color='tab:blue', label='Measured')
    if K ==  884: plt.plot(WL_1768, coefs[884:], color='tab:red' , label='Random')

    if K ==  884: plt.xticks(wl_1768, np.concatenate([wl_0884[::2], wl_0884[::2]]))

    if K ==  884: plt.axvline(WL_0884[-1], color='black', linestyle='--')
    if K == 1899: pass  # TODO

    plt.legend()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('UVE Coefficients / Stability')
    plt.title('%s UVE Coefficients' % title, fontproperties=prop)

    if save: plt.savefig('figure/%s' % save)
    if plot: plt.show()
    if True: plt.close()

    return coefs


def EEMD(X, Y, name):
    """
    多光谱 iPPG 信号获取与去噪算法
    Multi-spectral iPPG signal acquisition and denoising algorithm

    :param X: Original signal(s)
    :param Y: Measured / Reference label(s)
    :param name: Sample name / number (mainly for file saving purposes)
    :return: Processed signal(s)
    """
    N = X.shape[0]
    K = X.shape[1]

    colors = plt.get_cmap('autumn', N)
    x_ticks = range(0, K + 1)
    x_labels = ['%.2f' % (x_tick / 20) for x_tick in x_ticks[::10]]
    WLs = {0: '1100', 1: '1150', 2: '1200', 3: '1250', 4: '1300'}

    plt.figure(figsize=(24, 24))
    for i in range(N): plt.plot(x_ticks[1:], X[i] + (N - i - 1) * 600, color=colors(i), label='%s nm' % WLs[i])
    plt.xticks(x_ticks[::10], x_labels)
    plt.xlabel('Time (s)')
    plt.ylabel('相对光谱响应值')
    plt.legend(loc='upper right')
    plt.title('Sample %s: Time elapsed = %.2fs, heart rate = %d/min, SpO$_2$ = %d%%' % (name, K / 20, Y[0], Y[1]))
    plt.savefig('time/figure/original_signals_%s' % name)
    plt.close()

    """
    集合经验模态分解
    Ensemble Empirical Mode Decomposition (EEMD)
    https://pyemd.readthedocs.io/en/latest/examples.html
    """
    for i in range(N):

        # X[i] = savgol_filter(X[i], 15, 5)

        eemd = PyEMD.EEMD()
        eIMFs = eemd.eemd(X[i])
        nIMFs = eIMFs.shape[0]

        X[i] = eIMFs[1]

        fig, axs = plt.subplots(nIMFs + 2, 1, figsize=(24, 24))

        for ax in axs: ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%+g'))

        axs[0].plot(X[i])
        axs[0].locator_params(axis='y', nbins=2)
        axs[0].set_title('去除高频噪声后的信号', fontproperties=prop, fontsize=24)

        for n in range(nIMFs):

            axs[n + 1].plot(eIMFs[n])
            axs[n + 1].set_ylabel('IMF %i' % (n + 1))
            axs[n + 1].locator_params(axis='y', nbins=2)

        axs[-1].plot(X[i] - np.sum(eIMFs, axis=0))
        axs[-1].set_ylabel('Res')
        axs[-1].locator_params(axis='y', nbins=2)

        plt.tight_layout()
        plt.savefig('time/figure/EEMD_IMFs_%s' % name)
        plt.close()

    return X