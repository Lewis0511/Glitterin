"""
main.py
Author: Ziyang Liu @ Glitterin
Updated 2024.05.10
"""
import itertools
import utils

import numpy  as np
import pandas as pd

from collections import defaultdict
from matplotlib  import font_manager
from matplotlib  import pyplot as plt

from sklearn.cross_decomposition import PLSRegression

n_components = 2
prop = font_manager.FontProperties()

wl_0324 = np.arange(324)
wl_0884 = pd.read_excel('wl_0884.xlsx', 0, header=None).iloc[:, 0].to_numpy().astype(np.float64)  # [1240, 1700] nm
wl_1899 = pd.read_excel('wl_1899.xlsx', 1, header=None).iloc[:, 3].to_numpy().astype(np.float64)  # [ 867, 2530] nm

ticks_0324 = np.arange(0, 325, 18)
ticks_0884 = [int(wl) for wl in wl_0884[np.concatenate([np.arange(0,  883,  50),  [883]])]]
ticks_1899 = [int(wl) for wl in wl_1899[np.concatenate([np.arange(0, 1899, 100), [1898]])]]


def plot_correlation(data, name, names):

    X0, X1 = data[name][names[0]], data[name][names[1]]
    Y0, Y1 = data[name][names[2]], data[name][names[3]]

    if len(Y0.shape) == 1: Y0 = np.expand_dims(Y0, -1)
    if len(Y1.shape) == 1: Y1 = np.expand_dims(Y1, -1)

    K = X0.shape[1]
    M = Y0.shape[1]

    with np.errstate(invalid='ignore'):
        corr0 = np.array([np.corrcoef(X0.T[xi], Y0.T[yi])[0, 1] for xi, yi in zip(range(K), range(M) if M != 1 else [0] * K)])
        corr1 = np.array([np.corrcoef(X1.T[xi], Y1.T[yi])[0, 1] for xi, yi in zip(range(K), range(M) if M != 1 else [0] * K)])

    plt.figure(figsize=(24, 8))
    plt.plot(wl_0324 if K == 324 else wl_0884, corr0, label='with background')
    plt.plot(wl_0324 if K == 324 else wl_0884, corr1, label='  no background')
    plt.xticks(ticks_0324 if K == 324 else ticks_0884)
    plt.xlabel('PD Channels' if K == 324 else 'Wavelength (nm)')
    plt.ylabel('Correlation')
    plt.legend(loc='upper right')
    plt.title('%s: Correlation between %s and %s' % (name, names[0], names[2]))
    plt.savefig('lactate/figure/%s_correlation_%s_%s' % (name, names[0], names[2]))
    plt.close()

    return


def glucose(file_names, verbose=False):

    data = defaultdict(lambda: defaultdict(list))

    if verbose: print('\nSTART READING GLUCOSE DATA\n')

    for file_name in file_names:

        df = pd.read_excel('glucose/data/%s.xlsx' % file_name).to_numpy().astype(np.float64)
        for sample in df: data[file_name][sample[0]].append(sample[1:])
        if verbose: print('file_name = %s, data.shape = (%2d, %2d, %4d)' % (file_name, len(data[file_name]), len(data[file_name][0]), len(data[file_name][0][0])))

    if verbose: print('\nFINISH READING GLUCOSE DATA\n')

    return


def lactate(file_names, x_names, y_names, verbose=False):

    data = defaultdict(lambda: defaultdict())

    if verbose: print('\nSTART READING LACTATE DATA\n')

    for file_name in file_names:

        name = file_name[7:]
        for sheet_name in range(6): data[name][x_names[sheet_name]] = pd.read_excel('lactate/data/%s.xlsx' % file_name, sheet_name, header=None).to_numpy().astype(np.float64).T
        for column_idx in range(6): data[name][y_names[column_idx]] = pd.read_excel('lactate/data/%s.xlsx' % file_name, 6).iloc[:, column_idx + (name == '胡琮浩')].to_numpy()
        print('name = %s, data.shape =' % (name + (len(name) == 2) * '　'), [val.shape for val in data[name].values()])

    data['周欣雨'][y_names[4]][12] *= 10  # typo correction

    if verbose: print('\nFINISH READING LACTATE DATA\n')

    for name in data.keys():

        if verbose: print('\nSTART ANALYZING LACTATE DATA FOR %s' % name)

        for i, j in [[3, 4], [3, 5], [4, 5]]:
            print('Correlation between %s and %s: %+.4f%s' % (y_names[i], y_names[j],
                float(np.corrcoef(data[name][y_names[i]], data[name][y_names[j]])[0, 1]), '\n' * ([i, j] == [4, 5])))

        x_names.append('pd_sample_no_background')
        x_names.append('pd_source_no_background')

        data[name][x_names[7]] = data[name]['pd_sample'] - data[name]['pd_background']
        data[name][x_names[8]] = data[name]['pd_source'] - data[name]['pd_background']

        plot_correlation(data, name, [x_names[i] for i in [0, 0, 1, 7]])
        plot_correlation(data, name, [x_names[i] for i in [0, 0, 2, 8]])
        plot_correlation(data, name, [x_names[i] for i in [1, 7, 2, 8]])
        plot_correlation(data, name, [x_names[i] for i in [4, 3, 5, 5]])

        plot_correlation(data, name, [x_names[1], x_names[7], y_names[3], y_names[3]])
        plot_correlation(data, name, [x_names[1], x_names[7], y_names[4], y_names[4]])
        plot_correlation(data, name, [x_names[1], x_names[7], y_names[5], y_names[5]])

        plot_correlation(data, name, [x_names[4], x_names[3], y_names[3], y_names[3]])
        plot_correlation(data, name, [x_names[4], x_names[3], y_names[4], y_names[4]])
        plot_correlation(data, name, [x_names[4], x_names[3], y_names[5], y_names[5]])

        Y = pls_model_1 = None  # to shut up the inspections

        for [xi0, xi1], yi in itertools.product([[1, 7], [4, 3]], [3, 4, 5]):

            X0 = data[name][x_names[xi0]]
            X1 = data[name][x_names[xi1]]
            Y  = data[name][y_names[yi]]

            pls_model_0 = PLSRegression(n_components, scale=False)
            pls_model_1 = PLSRegression(n_components, scale=False)

            pls_model_0.fit(X0, Y)
            pls_model_1.fit(X1, Y)

            Y_pred_0 = pls_model_0.predict(X0).squeeze()
            Y_pred_1 = pls_model_1.predict(X0).squeeze()

            utils.plot_prediction_pls_and_pcr(Y, Y_pred_0, Y_pred_1, Y, Y_pred_0, Y_pred_1, None, None,
                '%s: %s vs. %s PLS Model, n_components = %d, with / no background' % (name, x_names[xi0], y_names[yi], n_components),
                'lactate/figure/%s_pls_pls_predictions_%s_%s' % (name, x_names[xi0], y_names[yi]), False)

        pls_model = pls_model_1  # pls_model is that for recon_sample_no_background vs. lactic_acid for now

        """ FIG 0. PLS COEFFICIENTS """
        plt.figure(figsize=(24, 8))
        plt.plot(wl_0884, pls_model.coef_.squeeze())
        plt.xticks(ticks_0884)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('PLS Coefficients')
        plt.title('%s Fig 0. The PLS Coefficients.' % name)
        plt.savefig('lactate/figure/%s_fig_00' % name)
        plt.close()

        """ FIG 1. PLS X LOADINGS P1 AND P2 """
        plt.figure(figsize=(24, 8))
        plt.plot(wl_0884, pls_model.x_loadings_[:, 0], label='$\mathbf{p}_1$')
        plt.plot(wl_0884, pls_model.x_loadings_[:, 1], label='$\mathbf{p}_2$')
        plt.xticks(ticks_0884)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('$x$-loadings')
        plt.legend()
        plt.title('%s Fig 1. The PLS $x$-loadings $\mathbf{p}_1$ and $\mathbf{p}_2$.' % name)
        plt.savefig('lactate/figure/%s_fig_01' % name)
        plt.close()

        min_score = max(min(pls_model.x_scores_[:, 0]), min(pls_model.y_scores_[:, 0]))
        max_score = min(max(pls_model.x_scores_[:, 0]), max(pls_model.y_scores_[:, 0]))
        ideal = np.linspace(min_score, max_score, 2)

        """ FIG 2. PLS X SCORE T1 AND Y SCORE U1 """
        plt.figure(figsize=(8, 8))
        plt.plot(ideal, ideal, alpha=0.5, color='k')
        plt.scatter(pls_model.x_scores_[:, 0], pls_model.y_scores_[:, 0], 4, (max(Y) - Y) / (max(Y) - min(Y)), cmap='seismic')
        plt.axhline(0, alpha=0.5, color='k')
        plt.axvline(0, alpha=0.5, color='k')
        plt.xlabel('$\mathbf{t}_1$')
        plt.ylabel('$\mathbf{u}_1$')
        plt.title('%s Fig 2. The PLS scores $\mathbf{t}_1$ and $\mathbf{u}_1$.' % name)
        plt.savefig('lactate/figure/%s_fig_02' % name)
        plt.close()

        """ FIG 3. PLS X SCORE T1 AND X SCORE T2 """
        plt.figure(figsize=(8, 8))
        plt.scatter(pls_model.x_scores_[:, 0], pls_model.x_scores_[:, 1], 4, (max(Y) - Y) / (max(Y) - min(Y)), cmap='seismic')
        plt.axhline(0, alpha=0.5, color='k')
        plt.axvline(0, alpha=0.5, color='k')
        plt.xlabel('$\mathbf{t}_1$')
        plt.ylabel('$\mathbf{t}_2$')
        plt.title('%s Fig 3. The PLS scores $\mathbf{t}_1$ and $\mathbf{t}_2$.' % name)
        plt.savefig('lactate/figure/%s_fig_03' % name)
        plt.close()

        if verbose: print('\nFINISH ANALYZING LACTATE DATA FOR %s' % name)

    return


def plastic(file_names, verbose=False):

    data = defaultdict(list)

    if verbose: print('\nSTART READING PLASTIC DATA\n')

    for file_name in file_names:

        df = pd.read_excel('plastic/data/%s.xlsx' % file_name).to_numpy()
        for sample in df: data[sample[0]].append(sample[1:].astype(np.float64))
        for key, val in data.items(): print('name = %4s, data.shape = (%d, %d)' % (key, len(val), len(val[0])))

    if verbose: print('\nFINISH READING PLASTIC DATA\n')

    return


""" PLEASE, PLEASE, PLEASE, TRY YOUR BEST TO MAKE TEMPORARY MODIFICATIONS ONLY IN MAIN() """


def main(prop_size=16, prop_family=('DejaVu Sans', 'SimHei')):

    """ https://stackoverflow.com/questions/65493638/glyph-23130-missing-from-current-font """
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # show Chinese label
    plt.rcParams['axes.unicode_minus'] = False    # these two lines need to be set manually

    prop.set_size(prop_size)
    prop.set_family(prop_family)

    """ GLUCOSE """
    file_names = ['0100_absorbance', '2000_absorbance', '0250_transmittance', '2000_transmittance']
    glucose(file_names, verbose=True)

    """ LACTATE """
    file_names = ['240424_汪文东', '240428_周欣雨', '240428_杨森', '240506_胡琮浩']
    x_names = ['pd_background', 'pd_sample', 'pd_source', 'recon_sample_no_background', 'recon_sample', 'recon_source', 'labels']
    y_names = ['id', 'time', 'heart_rate_after_exercise', ' heart_rate', 'blood_sugar', 'lactic_acid']
    lactate(file_names, x_names, y_names, verbose=True)

    """ PLASTIC """
    file_names = ['plastic']
    plastic(file_names, verbose=True)

    return


if __name__ == '__main__': main()