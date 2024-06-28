"""
main.py
Author: Ziyang Liu @ Glitterin
Updated 2024.06.28
"""

import itertools
import utils

import numpy  as np
import scipy  as sp
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


def plot_correlation(label, name, data, names, labels):  # TODO: change variable 'label' to a better name; add docstring

    X0, X1 = np.array(data[name][names[0]]), np.array(data[name][names[1]])
    Y0, Y1 = np.array(data[name][names[2]]), np.array(data[name][names[3]])

    if len(Y0.shape) == 1: Y0 = np.expand_dims(Y0, -1)
    if len(Y1.shape) == 1: Y1 = np.expand_dims(Y1, -1)

    K = X0.shape[1]
    M = Y0.shape[1]
    wl = {324: wl_0324, 884: wl_0884, 1899: wl_1899}[K]
    ticks = {324: ticks_0324, 884: ticks_0884, 1899: ticks_1899}[K]

    with np.errstate(invalid='ignore'):
        corr0 = np.array([np.corrcoef(X0.T[xi], Y0.T[yi])[0, 1] for xi, yi in zip(range(K), range(M) if M != 1 else [0] * K)])
        corr1 = np.array([np.corrcoef(X1.T[xi], Y1.T[yi])[0, 1] for xi, yi in zip(range(K), range(M) if M != 1 else [0] * K)])

    plt.figure(figsize=(24, 8))
    plt.plot(wl, corr0, label=labels[0])
    plt.plot(wl, corr1, label=labels[1])
    plt.xticks(ticks)
    plt.xlabel('PD Channels' if K == 324 else 'Wavelength (nm)')
    plt.ylabel('Correlation')
    plt.legend(loc='upper right')
    plt.title('%s %s: correlation between %s and %s' % (label, name, names[0], names[2]))
    plt.savefig('%s/figure/%s_correlation_%s_%s' % (label, name, names[0], names[2]))
    plt.close()

    return


def plot_figure(label, Y, pls_model, name, fig_num):

    """ FIG 0. PLS COEFFICIENTS """
    if fig_num == 0:
        plt.figure(figsize=(24, 8))
        plt.plot(wl_0884, pls_model.coef_.squeeze())
        plt.xticks(ticks_0884)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('PLS Coefficients')
        plt.title('%s Fig 0. The PLS Coefficients.' % name)
        plt.savefig('%s/figure/%s_fig_00' % (label, name))
        plt.close()

    """ FIG 1. PLS X LOADINGS P1 AND P2 """
    if fig_num == 1:
        plt.figure(figsize=(24, 8))
        plt.plot(wl_0884, pls_model.x_loadings_[:, 0], label='$\mathbf{p}_1$')
        plt.plot(wl_0884, pls_model.x_loadings_[:, 1], label='$\mathbf{p}_2$')
        plt.xticks(ticks_0884)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('$x$-loadings')
        plt.legend()
        plt.title('%s Fig 1. The PLS $x$-loadings $\mathbf{p}_1$ and $\mathbf{p}_2$.' % name)
        plt.savefig('%s/figure/%s_fig_01' % (label, name))
        plt.close()

    min_score = max(min(pls_model.x_scores_[:, 0]), min(pls_model.y_scores_[:, 0]))
    max_score = min(max(pls_model.x_scores_[:, 0]), max(pls_model.y_scores_[:, 0]))
    ideal = np.linspace(min_score, max_score, 2)

    """ FIG 2. PLS X SCORE T1 AND Y SCORE U1 """
    if fig_num == 2:
        plt.figure(figsize=(8, 8))
        plt.plot(ideal, ideal, alpha=0.5, color='k')
        plt.scatter(pls_model.x_scores_[:, 0], pls_model.y_scores_[:, 0], 4, (max(Y) - Y) / (max(Y) - min(Y)), cmap='seismic')
        plt.axhline(0, alpha=0.5, color='k')
        plt.axvline(0, alpha=0.5, color='k')
        plt.xlabel('$\mathbf{t}_1$')
        plt.ylabel('$\mathbf{u}_1$')
        plt.title('%s Fig 2. The PLS scores $\mathbf{t}_1$ and $\mathbf{u}_1$.' % name)
        plt.savefig('%s/figure/%s_fig_02' % (label, name))
        plt.close()

    """ FIG 3. PLS X SCORE T1 AND X SCORE T2 """
    if fig_num == 3:
        plt.figure(figsize=(8, 8))
        plt.scatter(pls_model.x_scores_[:, 0], pls_model.x_scores_[:, 1], 4, (max(Y) - Y) / (max(Y) - min(Y)), cmap='seismic')
        plt.axhline(0, alpha=0.5, color='k')
        plt.axvline(0, alpha=0.5, color='k')
        plt.xlabel('$\mathbf{t}_1$')
        plt.ylabel('$\mathbf{t}_2$')
        plt.title('%s Fig 3. The PLS scores $\mathbf{t}_1$ and $\mathbf{t}_2$.' % name)
        plt.savefig('%s/figure/%s_fig_03' % (label, name))
        plt.close()

    # TODO: KEEP ADDING FIGURES MY BRO

    return


def glucose(file_names, verbose=False):

    data = defaultdict(lambda: defaultdict(list))

    if verbose: print('\nSTART READING GLUCOSE DATA\n')

    for file_name in file_names:

        df = pd.read_excel('glucose/data/%s.xlsx' % file_name).to_numpy().astype(np.float64)
        for sample in df:
            data[file_name[5:]][file_name[:4] + '_glucose'].append(sample[0])
            data[file_name[5:]][file_name[:4] + '_spectra'].append(sample[1:])

    if verbose:
        for spectra_type in ['absorbance', 'transmittance']:
            for name, values in data[spectra_type].items(): print('spectra_type: %s, name: %s, values.shape:' % (spectra_type, name.ljust(12)), np.array(values).shape)

    if verbose: print('\nFINISH READING GLUCOSE DATA\n')

    plot_correlation('glucose',    'absorbance', data, ['0100_spectra', '2000_spectra', '0100_glucose', '2000_glucose'], ['01.00%', '20.00%'])
    plot_correlation('glucose', 'transmittance', data, ['0250_spectra', '2000_spectra', '0250_glucose', '2000_glucose'], ['02.50%', '20.00%'])

    for spectra_type, levels in [['absorbance', ['0100', '2000']], ['transmittance', ['0250', '2000']]]:

        if verbose: print('\nSTART ANALYZING GLUCOSE DATA FOR %s\n' % spectra_type.upper())

        # TODO: convert the following part into a function to increase code re-usability

        X0, Y0 = data[spectra_type][levels[0] + '_spectra'], data[spectra_type][levels[0] + '_glucose']
        X1, Y1 = data[spectra_type][levels[1] + '_spectra'], data[spectra_type][levels[1] + '_glucose']

        X0, Y0 = np.array(X0), np.array(Y0)
        X1, Y1 = np.array(X1), np.array(Y1)

        pls_model_0 = PLSRegression(n_components)
        pls_model_1 = PLSRegression(n_components)

        pls_model_0.fit(X0, Y0)
        pls_model_1.fit(X1, Y1)

        # Y_pred_0 = pls_model_0.predict(X0).squeeze()
        # Y_pred_1 = pls_model_1.predict(X1).squeeze()

        if verbose: print('\nFINISH ANALYZING GLUCOSE DATA FOR %s\n' % spectra_type.upper())

    return


def lactate(file_names, x_names, y_names, verbose=False):

    data = defaultdict(lambda: defaultdict())

    if verbose: print('\nSTART READING LACTATE DATA\n')

    for file_name in file_names:

        name = file_name[7:]
        for sheet_name in range(6): data[name][x_names[sheet_name]] = pd.read_excel('lactate/data/%s.xlsx' % file_name, sheet_name, header=None).to_numpy().astype(np.float64).T
        for column_idx in range(6): data[name][y_names[column_idx]] = pd.read_excel('lactate/data/%s.xlsx' % file_name, 6).iloc[:, column_idx + (name == '胡琮浩')].to_numpy()
        if verbose: print('name = %s, data.shape =' % (name + (len(name) == 2) * '　'), [val.shape for val in data[name].values()])

    data['周欣雨'][y_names[4]][12] *= 10  # typo correction

    if verbose: print('\nFINISH READING LACTATE DATA\n')

    for name in data.keys():

        if verbose: print('\nSTART ANALYZING LACTATE DATA FOR %s\n' % name)

        if verbose:
            for i, j in [[3, 4], [3, 5], [4, 5]]:
                print('Correlation between %s and %s: %+.4f%s' % (y_names[i], y_names[j],
                    float(np.corrcoef(data[name][y_names[i]], data[name][y_names[j]])[0, 1]), '\n' * ([i, j] == [4, 5])))

        x_names.append('pd_sample_no_background')
        x_names.append('pd_source_no_background')

        data[name][x_names[7]] = data[name]['pd_sample'] - data[name]['pd_background']
        data[name][x_names[8]] = data[name]['pd_source'] - data[name]['pd_background']

        labels = ['with_background', '  no_background']

        plot_correlation('lactate', name, data, [x_names[i] for i in [0, 0, 1, 7]], labels)
        plot_correlation('lactate', name, data, [x_names[i] for i in [0, 0, 2, 8]], labels)
        plot_correlation('lactate', name, data, [x_names[i] for i in [1, 7, 2, 8]], labels)
        plot_correlation('lactate', name, data, [x_names[i] for i in [4, 3, 5, 5]], labels)

        plot_correlation('lactate', name, data, [x_names[1], x_names[7], y_names[3], y_names[3]], labels)
        plot_correlation('lactate', name, data, [x_names[1], x_names[7], y_names[4], y_names[4]], labels)
        plot_correlation('lactate', name, data, [x_names[1], x_names[7], y_names[5], y_names[5]], labels)

        plot_correlation('lactate', name, data, [x_names[4], x_names[3], y_names[3], y_names[3]], labels)
        plot_correlation('lactate', name, data, [x_names[4], x_names[3], y_names[4], y_names[4]], labels)
        plot_correlation('lactate', name, data, [x_names[4], x_names[3], y_names[5], y_names[5]], labels)

        Y = pls_model_1 = None  # to shut up the inspections

        for [xi0, xi1], yi in itertools.product([[1, 7], [4, 3]], [3, 4, 5]):

            X0 = data[name][x_names[xi0]]
            X1 = data[name][x_names[xi1]]
            Y  = data[name][y_names[yi]]

            pls_model_0 = PLSRegression(n_components)
            pls_model_1 = PLSRegression(n_components)

            pls_model_0.fit(X0, Y)
            pls_model_1.fit(X1, Y)

            Y_pred_0 = pls_model_0.predict(X0).squeeze()
            Y_pred_1 = pls_model_1.predict(X1).squeeze()

            utils.plot_prediction_pls_and_pcr(Y, Y_pred_0, Y_pred_1, Y, Y_pred_0, Y_pred_1, None, None,
                '%s: %s vs. %s PLS model, n_components = %d, with / no background' % (name, x_names[xi0], y_names[yi], n_components),
                'lactate/figure/%s_pls_predictions_%s_%s' % (name, x_names[xi0], y_names[yi]), False)

        pls_model = pls_model_1  # pls_model is that for recon_sample_no_background versus lactic_acid for now

        for fig_num in range(4): plot_figure('lactate', Y, pls_model, name, fig_num)

        if verbose: print('\nFINISH ANALYZING LACTATE DATA FOR %s\n' % name)

    return


def moisture(file_names, verbose=False):

    data = defaultdict(lambda: defaultdict())

    if verbose: print('\nSTART READING MOISTURE DATA\n')

    for file_name in file_names:

        df0 = pd.read_excel('moisture/data/%s.xlsx' % file_name, 0, header=None)
        df1 = pd.read_excel('moisture/data/%s.xlsx' % file_name, 1, header=None)
        df2 = pd.read_excel('moisture/data/%s.xlsx' % file_name, 2)

        data[file_name]['sample'] = df0.to_numpy().astype(np.float64)
        data[file_name]['source'] = df1.to_numpy().astype(np.float64)
        data[file_name]['label'] = df2.iloc[:, 1].to_numpy()

        if verbose: print(file_name, [data[file_name][name].shape for name in ['sample', 'source', 'label']])

    if verbose: print('\nFINISH READING MOISTURE DATA\n')

    for file_name in file_names:

        if verbose: print('\nSTART ANALYZING MOISTURE DATA FOR %s' % file_name)

        plot_correlation('moisture', file_name, data, ['sample', 'source', 'label', 'label'], ['sample', 'source'])

        Y = data[file_name]['label']

        for X0, X1 in [[data[file_name]['sample'], data[file_name]['source']]]:

            pls_model_0 = PLSRegression(n_components)
            pls_model_1 = PLSRegression(n_components)

            pls_model_0.fit(X0, Y)
            pls_model_1.fit(X1, Y)

            Y_pred_0 = pls_model_0.predict(X0).squeeze()
            Y_pred_1 = pls_model_1.predict(X1).squeeze()

            utils.plot_prediction_pls_and_pcr(Y, Y_pred_0, Y_pred_1, Y, Y_pred_0, Y_pred_1, None, None,
                '%s: sample / source vs. skin moisture PLS model' % file_name,
                'moisture/figure/%s_pls_predictions_sample_source' % file_name, False)

        pls_model = pls_model_0  # pls_model is that for sample versus skin moisture for now

        for fig_num in range(4): plot_figure('moisture', Y, pls_model, file_name, fig_num)

        if verbose: print('\nFINISH ANALYZING MOISTURE DATA FOR %s' % file_name)

    return


def plastic(file_names, verbose=False):

    data = defaultdict(list)

    if verbose: print('\nSTART READING PLASTIC DATA\n')

    for file_name in file_names:

        df = pd.read_excel('plastic/data/%s.xlsx' % file_name).to_numpy()
        for sample in df: data[sample[0]].append(sample[1:].astype(np.float64))
        if verbose:
            for key, val in data.items(): print('name = %4s, data.shape = (%d, %d)' % (key, len(val), len(val[0])))

    if verbose: print('\nFINISH READING PLASTIC DATA\n')

    return


def time(file_names, verbose=False):

    data = defaultdict(lambda: defaultdict())
    WL = {0: '1100', 1: '1150', 2: '1200', 3: '1250', 4: '1300'}

    if verbose: print('\nSTART READING TIME DATA\n')

    for file_name in file_names:

        data[file_name]['labels'] = pd.read_excel('time/data/%s.xlsx' % file_name, '实测值').to_numpy().astype(np.float64)

        for i, sheet_name in enumerate(['1100nm', '1150nm', '1200nm', '1250nm', '1300nm']):
            data[file_name][WL[i]] = []
            df = pd.read_excel('time/data/%s.xlsx' % file_name, sheet_name).iloc[:, 1::2].T
            for row in range(len(df)): data[file_name][WL[i]].append(df.iloc[row, :].dropna().to_numpy().astype(np.float64))

    if verbose: print('\nFINISH READING TIME DATA\n')

    for file_name in file_names:

        if verbose: print('\nSTART ANALYZING TIME DATA FOR %s.xlsx\n' % file_name)

        Y_ref  = []
        Y_pred = []
        N = len(data[file_name]['labels'])
        T = 10.00

        for i in range(N):

            X = np.array([utils.mean_centering(data[file_name][WL[wl]][i]) for wl in range(5)])
            Y = data[file_name]['labels'][i]
            K = len(X[0])

            x_eemd = utils.EEMD(X, Y, '%s_%02d' % (file_name, i))
            x_fft = sp.fft.fft(x_eemd[0])[: K // 2]

            plt.figure(figsize=(16, 8))
            plt.plot([t / 60 * T for t in range(K // 2)], np.abs(x_fft), color='blue')
            plt.xticks([t / 60 * T for t in range(0, K // 2, 3)])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Normalized Amplitude')
            plt.title('Sample %02d: Fast Fourier Transform Spectrum of Signal' % i)
            plt.savefig('time/figure/%s_%02d_fast_fourier_transform' % (file_name, i))
            plt.close()

            Y_ref .append(Y[0])
            Y_pred.append(np.argmax(np.abs(x_fft)) * 6.0)

        print('Reference heart rate:', [int(y) for y in Y_ref ])
        print('Predicted heart rate:', [int(y) for y in Y_pred])
        utils.plot_prediction(Y_ref, Y_pred, Y_ref, Y_pred, None, None,
                              '%s Heart Rate' % file_name, 'time/figure/%s_prediction' % file_name, False)

        if verbose: print('\nFINISH ANALYZING TIME DATA FOR %s.xlsx\n' % file_name)

    return


""" PLEASE, PLEASE, PLEASE, TRY YOUR BEST TO MAKE TEMPORARY MODIFICATIONS ONLY IN MAIN() """


def main(GLUCOSE=True, LACTATE=True, MOISTURE=True, PLASTIC=True, TIME=True, prop_size=16, prop_family=('DejaVu Sans', 'SimHei')):

    """ https://stackoverflow.com/questions/65493638/glyph-23130-missing-from-current-font """
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # show Chinese label
    plt.rcParams['axes.unicode_minus'] = False    # these two lines need to be set manually

    prop.set_size(prop_size)
    prop.set_family(prop_family)

    """ GLUCOSE """
    file_names = ['0100_absorbance', '2000_absorbance', '0250_transmittance', '2000_transmittance']
    if GLUCOSE: glucose(file_names, verbose=True)

    """ LACTATE """
    file_names = ['240424_汪文东', '240428_周欣雨', '240428_杨森', '240506_胡琮浩']
    x_names = ['pd_background', 'pd_sample', 'pd_source', 'recon_sample_no_background', 'recon_sample', 'recon_source', 'labels']
    y_names = ['id', 'time', 'heart_rate_after_exercise', ' heart_rate', 'blood_sugar', 'lactic_acid']
    if LACTATE: lactate(file_names, x_names, y_names, verbose=True)

    """ MOISTURE """
    file_names = ['皮肤水分-样机']
    if MOISTURE: moisture(file_names, verbose=True)

    """ PLASTIC """
    file_names = ['plastic']
    if PLASTIC: plastic(file_names, verbose=True)

    """ TIME """
    file_names = ['0528', '指尖0529', '脉搏0529']
    if TIME: time(file_names, verbose=True)

    return


if __name__ == '__main__': main(True, True, True, True, True)