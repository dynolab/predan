import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime as dt
import itertools as it
import scipy as sp
from itertools import combinations
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split


# Several useful features
def generate_ts(date_start: str = "29/06/2023 00:00:00", date_end: str = "30/07/2023 00:00:00", alpha: float = 0.6,
                beta: float = 0.4, max_kf: float = 5, trend: float = 4, season: float = 0) -> pd.Series:
    """
    Generate time series with autocorrelation, trend and seasons dependency.
    Parameters:
        date_start(str): first data in time series in 'DD/MM/YYYY HH:MM:SS' format
        date_end(str): last data in time series in 'DD/MM/YYYY HH:MM:SS' format
        alpha(float): correlation coefficient with previous value
        beta(float): correlation coefficient with previous last value
        max_kf(float): maximum absolute value of the additional noise component with uniform distribution
        trend(float): absolute additional value for introducing a trend component
        season(float): absolute addition value for elements that are one week apart
    Returns:
        time_series: generated time series
    """
    time_series = pd.date_range(start=date_start, end=date_end, freq='H').to_series()
    time_series.index = time_series
    time_series[0] = 5000.0
    time_series[1] = 5015.0
    date = 2
    for i in time_series[2:]:
        time_series[i] = (alpha * time_series[i - dt.timedelta(1 / 24)] + beta * time_series[
            i - dt.timedelta(2 / 24)]) + np.random.uniform(-max_kf, max_kf) + trend
        if date % (7 * 24) == 0:
            time_series[i] += season
        if date % (7 * 24) == 1:
            time_series[i] -= season
        date += 1
    return time_series


def random_walk(date_start: str = "29/06/2023 00:00:00", date_end: str = "30/07/2023 00:00:00", noise: float = 10,
                noise_dist: str = 'normal') -> pd.Series:
    """
    Generate time series as random walk.
    Parameters:
        date_start(str): first data in time series in 'DD/MM/YYYY HH:MM:SS' format
        date_end(str): last data in time series in 'DD/MM/YYYY HH:MM:SS' format
        noise(float): noise component
        noise_dist(str): distribution of noise. 'normal': np.random.normal(0,noise). 'uniform': np.random.uniform(-noise, noise)
    Returns:
        time_series: generated time series
    """
    time_series = pd.date_range(start=date_start, end=date_end, freq='H').to_series()
    time_series.index = time_series
    time_series[date_start] = 5000.0
    if noise_dist != 'uniform':
        for i in time_series[1:]:
            time_series[i] = time_series[i - dt.timedelta(1 / 24)] + np.random.normal(0, noise)
    else:
        for i in time_series[1:]:
            time_series[i] = time_series[i - dt.timedelta(1 / 24)] + np.random.uniform(-noise, noise)
    return time_series


def equals_test(n1: int, n2: int, n: int):
    """
    Test of two value of permutations equally likely
    Parameters:
        n1(int): number of objects with the first property
        n2(int): number of objects with the second property
        n(int): number of all objects in sample
    Returns:
        z, p_value((float,float)): statistics and p_value of test
    """
    w1 = n1 / n
    w2 = n2 / n
    p = (n1 + n2) / (2 * n)
    z = 0
    if p != 0 and p != 1:
        z = (w1 - w2) / (p * (1 - p) * (2 / n)) ** 0.5
    mod_z = abs(z)
    p_value = 2 * (1 - sp.stats.norm.cdf(mod_z))
    return z, p_value


def multiply_equals_test(permutations: list, alpha: float = 0.05, method: str = "hs"):
    """
    Symmetry category test with multiple hypothesis of equals tests
    Parameters:
        permutations(list): vector with a number of permutations of different types
        alpha(float): overall significance level
        method(str): method to multipletest
            bonferroni : one-step correctio
            sidak : one-step correction
            holm-sidak : step down method using Sidak adjustments
            holm : step-down method using Bonferroni adjustments
            simes-hochberg : step-up method (independent)
            hommel : closed method based on Simes tests (non-negative)
            fdr_bh : Benjamini/Hochberg (non-negative)
            fdr_by : Benjamini/Yekutieli (negative)
            fdr_tsbh : two stage fdr correction (non-negative)
            fdr_tsbky : two stage fdr correction (non-negative)
    Returns:
        list of bool(true for hypothesis that can be rejected for given alpha), p-values corrected for multiple tests, corrected alpha for Sidak method, corrected alpha for Bonferroni method
    """
    p_list = []
    for pair_index in range(len(permutations) // 2):
        statistic, p_value = equals_test(permutations[pair_index], permutations[-1 - pair_index],
                                         sum(permutations))
        p_list.append(p_value)
    return sm.stats.multipletests(p_list, alpha, method)


def big_split(sym_test: str, columns: list or None = None, test_size: float = 0.05, random_state: int = 42):
    """
    Splits all time series into train and test with sym_test and without
    :param columns: list of time series features
    :param test_size: size of test dataframe
    :param random_state: set randomizer seed
    :return:
    """
    dfM = pd.read_csv("../src/feature/month.csv", index_col=0)
    dfY = pd.read_csv("../src/feature/year.csv", index_col=0)
    dfQ = pd.read_csv("../src/feature/quater.csv", index_col=0)
    dfD = pd.read_csv("../src/feature/day.csv", index_col=0)
    dfH = pd.read_csv("../src/feature/hour.csv", index_col=0)
    dfW = pd.read_csv("../src/feature/week.csv", index_col=0)
    df_res = pd.read_csv("all_results.csv", index_col=0)
    assert sym_test in df_res.columns, "sym_test not found. You can choose one from ['sym_test', 'sym_test1', 'sym_test2', 'sym_pv', 'sym_pv1'"

    def split(df, format):
        y = df_res.loc[df_res.index.str.startswith(format)]
        y = y.reindex(df.index)
        if columns is None:
            X_train, X_test, y_train, y_test = train_test_split(df, y['smape'], test_size=test_size,
                                                                random_state=random_state)
        else:
            X_train, X_test, y_train, y_test = train_test_split(df[columns], y['smape'], test_size=test_size,
                                                                random_state=random_state)
        df_with_test = df.copy()
        df_with_test[sym_test] = y[sym_test]
        if columns is None:
            X_trainT, X_testT, y_trainT, y_testT = train_test_split(df_with_test, y['smape'], test_size=test_size,
                                                                    random_state=random_state)
        else:
            X_trainT, X_testT, y_trainT, y_testT = train_test_split(df_with_test[columns + [sym_test]], y['smape'],
                                                                    test_size=test_size, random_state=random_state)
        return ([X_train, X_test, y_train, y_test], [X_trainT, X_testT, y_trainT, y_testT])

    df_splitY, df_splitYT = split(dfY, 'Y')
    df_splitQ, df_splitQT = split(dfQ, 'Q')
    df_splitM, df_splitMT = split(dfM, 'M')
    df_splitW, df_splitWT = split(dfW, 'W')
    df_splitD, df_splitDT = split(dfD, 'D')
    df_splitH, df_splitHT = split(dfH, 'H')
    X_train = pd.concat([df_splitY[0], df_splitQ[0], df_splitM[0], df_splitW[0], df_splitD[0], df_splitH[0]])
    X_test = pd.concat([df_splitY[1], df_splitQ[1], df_splitM[1], df_splitW[1], df_splitD[1], df_splitH[1]])
    y_train = pd.concat([df_splitY[2], df_splitQ[2], df_splitM[2], df_splitW[2], df_splitD[2], df_splitH[2]])
    y_test = pd.concat([df_splitY[3], df_splitQ[3], df_splitM[3], df_splitW[3], df_splitD[3], df_splitH[3]])
    X_trainT = pd.concat([df_splitYT[0], df_splitQT[0], df_splitMT[0], df_splitWT[0], df_splitDT[0], df_splitHT[0]])
    X_testT = pd.concat([df_splitYT[1], df_splitQT[1], df_splitMT[1], df_splitWT[1], df_splitDT[1], df_splitHT[1]])
    y_trainT = pd.concat([df_splitYT[2], df_splitQT[2], df_splitMT[2], df_splitWT[2], df_splitDT[2], df_splitHT[2]])
    y_testT = pd.concat([df_splitYT[3], df_splitQT[3], df_splitMT[3], df_splitWT[3], df_splitDT[3], df_splitHT[3]])
    return X_train, X_test, y_train, y_test, X_trainT, X_testT, y_trainT, y_testT


def generate_arma(p, q, n):
    """
    Генерация авторегрессионного скользящего временного ряда (ARMA(p,q)).

    Параметры:
    - p: int, порядок авторегрессии
    - q: int, порядок скользящего среднего
    - n: int, количество точек в ряду

    Возвращает:
    - временной ряд
    """
    arparams = np.random.uniform(low=-1, high=1, size=p)
    maparams = np.random.uniform(low=-1, high=1, size=q)
    ar = np.r_[1, -arparams]  # add zero-lag and negate
    ma = np.r_[1, maparams]  # add zero-lag
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    while np.min(np.abs(arma_process.arroots)) < 1:
        arparams = np.random.uniform(low=-1, high=1, size=p)
        maparams = np.random.uniform(low=-1, high=1, size=q)
        ar = np.r_[1, -arparams]  # add zero-lag and negate
        ma = np.r_[1, maparams]  # add zero-lag
        arma_process = sm.tsa.ArmaProcess(ar, ma)
    time_series = arma_process.generate_sample(n)
    return time_series


def generate_rw(n, noise_variance=1.0):
    """
    Генерация случайного блуждания.

    Параметры:
    - n: int, количество точек в ряду
    - noise_variance: float, дисперсия шума

    Возвращает:
    - временной ряд
    """
    # Генерация случайного шума с заданной дисперсией
    noise = np.random.normal(0, np.sqrt(noise_variance), size=n)

    # Создание временного ряда из случайного шума
    time_series = np.cumsum(noise)

    return time_series


def generate_multifractal(n, H):
    """
    Генерация мультифрактального временного ряда.

    Параметры:
    - n: int, количество точек в ряду
    - H: float, параметр Хёрста, определяющий степень мультифрактальности

    Возвращает:
    - временной ряд
    """
    # Генерация случайного шума
    noise = np.random.randn(n)

    # Создание временного ряда из случайного шума
    time_series = np.cumsum(noise)

    # Применение мультифрактальности к временному ряду
    time_series = time_series / np.std(time_series)
    time_series = np.power(np.abs(time_series), H) * np.sign(time_series)

    return time_series


def generate_trend_series(length=250, trend_koef=2, ampl=10, seasonality_period=7, seasonality_aplitude=1,
                          sigma=1):
    """
    Generate sin series with trend? seasonality and noise.
    :param length: length of the series.
    :param freq: frequency of the sin.
    :param ampl: amplitude of the sin.
    :param seasonality_period: The period of the seasonality (e.g., 12 for monthly data).
    :param seasonality_aplitude: The amplitude of the seasonality
    :param sigma: The standard deviation of the error term.
    :return: A NumPy array representing the sin series with trend, seasonality, and error.
    """

    def add_components(stationary_series, trend_type="linear", seasonality_period=7, seasonality_aplitude=1,
                       sigma=1, trend_koef=1):
        """
      This function adds trend, seasonality, and error to a stationary series.

      Args:
        stationary_series: A NumPy array representing the stationary series.
        trend_type: The type of trend to add. Can be "linear" or "nonlinear".
        seasonality_period: The period of the seasonality (e.g., 12 for monthly data).
        sigma: The standard deviation of the error term.
        seasonality_aplitude: The amplitude of the seasonality

      Returns:
          A NumPy array representing the series with trend, seasonality, and error.
      """

        # Add trend
        if trend_type == "linear":
            trend = np.arange(len(stationary_series))
        else:
            # TODO Add a more complex nonlinear trend if desired
            # ...
            pass

        # Add seasonality
        seasonality = np.zeros(len(stationary_series))
        for i in range(len(stationary_series)):
            seasonality[i] = seasonality_aplitude * np.sin(2 * np.pi * i / seasonality_period)

        # Add error
        error = np.random.normal(loc=0, scale=sigma, size=len(stationary_series))

        # Combine all components
        series_with_trend_seasonality = stationary_series + trend * trend_koef + seasonality
        series_with_trend_seasonality_error = series_with_trend_seasonality + error

        return series_with_trend_seasonality, series_with_trend_seasonality_error

    x = np.linspace(0, 2 * np.pi, length)
    series = ampl * np.sin(x)
    new_series, new_series_with_error = add_components(series, seasonality_period=seasonality_period,
                                seasonality_aplitude=seasonality_aplitude, sigma=sigma, trend_koef=trend_koef)
    return new_series, new_series_with_error

def arima_forecast(data, p, d, q, number_of_steps):
    model = ARIMA(data, order=(p, d, q))
    fitted_model = model.fit()
    forecasts = fitted_model.forecast(steps=number_of_steps)
    return forecasts

def ar2_forecast(data, number_of_steps):
    return arima_forecast(data, 2, 0, 0, number_of_steps)

def arma_forecast(data, number_of_steps):
    return arima_forecast(data, 10, 0, 5, number_of_steps)
def naive_forecast(data, number_of_steps):
    last_observation = data[-1]
    index = np.arange(data.shape[0], data.shape[0] + number_of_steps)
    forecast = pd.Series(last_observation, index=index)
    return forecast

def mean_absolute_scaled_error(insample, y_test, y_hat_test, freq):
    """
    Calculates MAsE

    :param insample: insample data
    :param y_test: out of sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency
    :return:
    """
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])

    masep = np.mean(abs(insample[freq:] - y_hat_naive))

    return np.mean(abs(y_test - y_hat_test)) / masep

def symmetric_mean_absolute_percentage_error(a, b):
    """
    Calculates sMAPE

    :param a: actual values
    :param b: predicted values
    :return: sMAPE
    """
    a = np.reshape(a, (-1,))
    b = np.reshape(b, (-1,))
    return np.mean(2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()