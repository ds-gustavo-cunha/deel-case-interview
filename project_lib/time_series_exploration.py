##############################
####### INITIAL CONFIG #######
##############################

# import required library to configure module
import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict
from project_lib.initial_config import initial_settings
from project_lib.input_validation import (
    validate_input_types,
    validate_dataframe_cols,
    validate_input_values,
)

# set the basic cofiguration for this module
initial_settings()


################################
####### MODULE FUNCTIONS #######
################################


def plot_pacf_acf(
    dataframe: pd.DataFrame,
    columns: List[str],
    figsize: Union[tuple, None] = None,
    max_lags: Union[int, None] = None,
    x_ticks: Union[list, None] = None,
) -> dict:
    """
    Plot PACF and ACF for the 'columns' input on the input dataframe.

    Args
        dataframe: a pd.DataFrame with columns to plot
        columns: a list with the names of columns to plot
        figsize: a tuple with figure size. If not input, it will be
            automatically defined based on number of columns to plot.
        max_lags: an int with maximum number of lags to plot.
            If not input, will be automatically defined.
        x_ticks: a list with x_tick on plot in case it makes easier
            to interpret results.

    Return
        dict_pacf: a dict with keys as columns and values as a dataframe
            with X and Y data from PACF plot
    """

    # input validation
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"columns": columns}, (list,))
    validate_dataframe_cols(
        dataframe=dataframe, columns=tuple(columns), error_msg="Incorrect column names!"
    )
    if figsize is not None:
        validate_input_types({"figsize": figsize}, (tuple,))
    if max_lags is not None:
        validate_input_types({"max_lags": max_lags}, (int,))
    if x_ticks is not None:
        validate_input_types({"x_ticks": x_ticks}, (list,))

    # import required libraries
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from matplotlib.gridspec import GridSpec

    # instanciate a dict to save plot results
    dict_pacf = dict()

    # define plot style
    plt.style.use("fivethirtyeight")

    # check if user input figsize
    if figsize is None:
        # assign th default figsize
        figsize = (15, len(columns) * 3)

    # define figure
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = GridSpec(nrows=len(columns), ncols=2, figure=fig)

    # iterate over columns to plot
    for idx, col in enumerate(columns):
        # plot acf and padf
        axs_acf = fig.add_subplot(gs[idx, 0])
        axs_pacf = fig.add_subplot(gs[idx, 1])
        # plot acf
        plot_acf(
            x=dataframe[col].dropna().values,
            title=f"ACF: {col}",
            auto_ylims=True,
            lags=max_lags,
            c="r",
            zero=True,
            ax=axs_acf,
        )
        # plot pacf
        plot_pacf(
            x=dataframe[col].dropna().values,
            title=f"PACF: {col}",
            auto_ylims=True,
            lags=max_lags,
            c="b",
            zero=True,
            ax=axs_pacf,
        )
        # get acf data
        ax_ = plt.gca()
        # save data on dict_plots
        dict_pacf[col] = pd.DataFrame(
            data=dict(
                x_pacf=ax_.lines[1].get_xdata(),
                y_pacf=ax_.lines[1].get_ydata()
            )
        )
        # define plot details
        for ax in [axs_acf, axs_pacf]:
            ax.set_ylim(ymax=1.05)
            if x_ticks is not None:
                ax.set_xticks(x_ticks)

    # display plot
    plt.show()

    return dict_pacf


def compare_ma_and_loess_seasonal_decompositions(
    dataframe: pd.DataFrame,
    columns: List[str],
    period: int,
    moving_avg_add_params: Union[dict, None] = dict(),
    loess_add_params: Union[dict, None] = dict(),
    figsize: tuple = (15, 10),
) -> None:
    """
    Plot moving average and LOESS seasonal decomposition given the input params.

    Args
        dataframe: a pd.DataFrame with columns to plot decomposition
        columns: a list with the name of columns to plot decomposition
        period: an int with the period param for decomposition
        moving_avg_add_params: a dict with params for statsmodels.tsa.seasonal.seasonal_decompose function
        loess_add_params: a dict with params for statsmodels.tsa.seasonal.seasonal_decompose function
        figsize: a tuple with figure size of plot for each column

    Return
        None
    """

    # input validation
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"columns": columns}, (list,))
    validate_dataframe_cols(
        dataframe=dataframe, columns=tuple(columns), error_msg="Incorrect column names!"
    )
    if period is not None:
        validate_input_types({"max_lags": period}, (int,))
    if figsize is not None:
        validate_input_types({"figsize": figsize}, (tuple,))
    validate_input_types({"moving_avg_add_params": moving_avg_add_params}, (dict,))
    validate_input_types({"loess_add_params": loess_add_params}, (dict,))
    if "x" in moving_avg_add_params.keys():
        raise ValueError(
            "'x' param must not be input on moving_avg_add_params. Input it by means of the 'dataframe' param."
        )
    if "period" in moving_avg_add_params.keys():
        raise ValueError(
            "'period' param must not be input on moving_avg_add_params. Input it by means of the 'period' param."
        )
    if "endog" in loess_add_params.keys():
        raise ValueError(
            "'endog' param must not be input on loess_add_params. Input it by means of the 'dataframe' param."
        )
    if "period" in loess_add_params.keys():
        raise ValueError(
            "'period' param must not be input on loess_add_params. Input it by means of the 'period' param."
        )

    # import required libraries
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose, STL

    # define plot style
    plt.style.use("fivethirtyeight")

    # iterate over columns to plot
    for col in columns:
        # define decomposition params for moving avg decomposition
        moving_avg_default_params = dict(
            x=dataframe[col],  # user input
            period=period,  # user input
            model="multiplicative",
            filt=None,
            two_sided=True,
            extrapolate_trend="freq",
        )
        # overwrite decomposition params with the inputed ones
        moving_avg_params = {
            k: (v if moving_avg_add_params.get(k) is None else moving_avg_add_params[k])
            for k, v in moving_avg_default_params.items()
        }

        # define decomposition params for moving avg decomposition
        loess_default_params = dict(
            endog=dataframe[col],  # user input
            period=period,  # user input
            seasonal=7,
            trend=None,
            low_pass=None,
            seasonal_deg=1,
            trend_deg=1,
            low_pass_deg=1,
            robust=False,
            seasonal_jump=1,
            trend_jump=1,
            low_pass_jump=1,
        )
        # overwrite decomposition params with the inputed ones
        loess_params = {
            k: (v if loess_add_params.get(k) is None else loess_add_params[k])
            for k, v in loess_default_params.items()
        }

        # perform seasonal decomposition using moving averages
        sd_ma_add = seasonal_decompose(**moving_avg_params)
        # perform seasonal decomposition using LOESS.
        sd_loess = STL(**loess_params).fit()

        # define plot
        f, axs = plt.subplots(
            nrows=4, ncols=2, figsize=figsize, constrained_layout=True
        )
        # define figure title
        f.suptitle(f"{col.upper()}\n")

        # iterate over decompositions to plot
        for idx, decomp in enumerate(
            zip([sd_ma_add, sd_loess], ["Moving average", "LOESS"])
        ):
            # plot
            axs[0, idx].plot(decomp[0].observed)
            axs[1, idx].plot(decomp[0].trend)
            axs[2, idx].plot(decomp[0].seasonal)
            axs[3, idx].scatter(x=decomp[0].resid.index, y=decomp[0].resid)
        # plot details
        for i in range(0, 3):
            axs[i, idx].set(xticklabels=[])  # remove the tick labels
            axs[i, idx].tick_params(bottom=False)  # remove the ticks
        axs[0, 0].set_title(f"Moving average")
        axs[0, 1].set_title(f"LOESS")
        axs[0, 0].set_ylabel(f"Observed")
        axs[1, 0].set_ylabel(f"Trend")
        axs[2, 0].set_ylabel(f"Seasonal")
        axs[3, 0].set_ylabel(f"Residuals")

        # display chart
        plt.show()


def compare_feature_decompositions(
    dataframe: pd.DataFrame,
    columns: List[str],
    period: int,
    loess_decomp: bool = True,
    moving_avg_add_params: Union[dict, None] = dict(),
    loess_add_params: Union[dict, None] = dict(),
    figsize: Union[tuple, None] = None,
) -> None:
    """
    Compare moving LOESS (or moving average) seasonal decomposition over diferent features
    given the input params.

    Args
        dataframe: a pd.DataFrame with columns to plot decomposition
        columns: a list with the name of columns to plot decomposition
        period: an int with the period param for decomposition
        loess_decomp: a bool to indicate of use LOESS decomposition [loess_decomp = True] or
            moving average decomposition [loess_decomp = False]
        moving_avg_add_params: a dict with params for statsmodels.tsa.seasonal.seasonal_decompose function
        loess_add_params: a dict with params for statsmodels.tsa.seasonal.STL function
        figsize: a tuple with figure size of plot for each column

    Return
        None
    """

    # input validation
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"columns": columns}, (list,))
    validate_dataframe_cols(
        dataframe=dataframe, columns=tuple(columns), error_msg="Incorrect column names!"
    )
    validate_input_types({"loess_decomp": loess_decomp}, (bool,))
    if period is not None:
        validate_input_types({"max_lags": period}, (int,))
    if figsize is not None:
        validate_input_types({"figsize": figsize}, (tuple,))
    validate_input_types({"moving_avg_add_params": moving_avg_add_params}, (dict,))
    validate_input_types({"loess_add_params": loess_add_params}, (dict,))
    if "x" in moving_avg_add_params.keys():
        raise ValueError(
            "'x' param must not be input on moving_avg_add_params. Input it by means of the 'dataframe' param."
        )
    if "period" in moving_avg_add_params.keys():
        raise ValueError(
            "'period' param must not be input on moving_avg_add_params. Input it by means of the 'period' param."
        )
    if "endog" in loess_add_params.keys():
        raise ValueError(
            "'endog' param must not be input on loess_add_params. Input it by means of the 'dataframe' param."
        )
    if "period" in loess_add_params.keys():
        raise ValueError(
            "'period' param must not be input on loess_add_params. Input it by means of the 'period' param."
        )

    # import required libraries
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose, STL
    from matplotlib.gridspec import GridSpec

    # define plot style
    plt.style.use("fivethirtyeight")

    # define default figsize in case input was none
    if figsize is None:
        figsize = (len(columns) * 6.5, 10)

    # define plot
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = GridSpec(nrows=4, ncols=len(columns), figure=fig)

    # iterate over columns to plot
    for idx, col in enumerate(columns):
        # check if decomposition is not LOESS
        if not loess_decomp:
            # define decomposition params for moving avg decomposition
            moving_avg_default_params = dict(
                x=dataframe[col],  # user input
                period=period,  # user input
                model="multiplicative",
                filt=None,
                two_sided=True,
                extrapolate_trend="freq",
            )
            # overwrite decomposition params with the inputed ones
            moving_avg_params = {
                k: (
                    v
                    if moving_avg_add_params.get(k) is None
                    else moving_avg_add_params[k]
                )
                for k, v in moving_avg_default_params.items()
            }

            # perform seasonal decomposition using moving averages
            sd = seasonal_decompose(**moving_avg_params)

        # LOESS decomposition
        else:
            # define decomposition params for moving avg decomposition
            loess_default_params = dict(
                endog=dataframe[col],  # user input
                period=period,  # user input
                seasonal=7,
                trend=None,
                low_pass=None,
                seasonal_deg=1,
                trend_deg=1,
                low_pass_deg=1,
                robust=False,
                seasonal_jump=1,
                trend_jump=1,
                low_pass_jump=1,
            )
            # overwrite decomposition params with the inputed ones
            loess_params = {
                k: (v if loess_add_params.get(k) is None else loess_add_params[k])
                for k, v in loess_default_params.items()
            }

            # perform seasonal decomposition using LOESS.
            sd = STL(**loess_params).fit()

        # plot
        axs_obs = fig.add_subplot(gs[0, idx])
        axs_trend = fig.add_subplot(gs[1, idx])
        axs_season = fig.add_subplot(gs[2, idx])
        axs_error = fig.add_subplot(gs[3, idx])
        axs_obs.plot(sd.observed)
        axs_trend.plot(sd.trend)
        axs_season.plot(sd.seasonal)
        axs_error.scatter(x=sd.resid.index, y=sd.resid)
        # plot details
        for ax in [axs_obs, axs_trend, axs_season]:
            ax.set(xticklabels=[])  # remove the tick labels
            ax.tick_params(bottom=False)  # remove the ticks
        axs_obs.set_title(f"{col}:\n{'LOESS' if loess_decomp else 'moving average'} decomposition")
        axs_obs.set_ylabel(f"Observed")
        axs_trend.set_ylabel(f"Trend")
        axs_season.set_ylabel(f"Seasonal")
        axs_error.set_ylabel(f"Residuals")

    # display chart
    plt.show()


def inspect_trigonometric_features(
    dataframe: pd.DataFrame,
    columns: List[str],
    min_period: int = 7,
    max_period: int = 366,
    mutual_info: bool = True,
    x_ticks: Union[list, None] = None,
    figsize: Union[tuple, None] = None,
) -> dict:
    """
    Plot potential impact of trigonometric features when modelling the target variable.
    'Potential impact' is measured as sqrt( sin(2*time*pi*period)^2 + cos( 2*time*pi*period)^2)
    and gives a initial indication about how good trigonometric feature engineering will perform.
    It assumes the estimated mutual information or a modelling with linear regression, 
    both on trigonometric features.

    Args
        dataframe: a pd.DataFrame with target variable as time-orded features.
        columns: a list of columns to inspect trigonometric features.
        min_period: an int with the initial period to experimentat
        max_period: an int with the final period to experiment
        mutual_info: a bool flag to indicate if power will be calculated with
            sklearn.feature_selection.mutual_info_regression [mutual_info = True] or 
            statsmodels.formula.api.ols [mutual_info = False]
        x_ticks: a list with ticks to display on plot in case it makes easier
            to interpret results
        figsize: a tuple with figure size for the whole plot

    Return
        dict_trig: a dict where keys are columns and values are
            trigonometrics "potential impact"
    """

    # input validation
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"columns": columns}, (list,))
    validate_dataframe_cols(
        dataframe=dataframe, columns=tuple(columns), error_msg="Incorrect column names!"
    )
    validate_input_types({"mutual_info": mutual_info}, (bool,))
    validate_input_types({"min_period": min_period}, (int,))
    validate_input_types({"max_period": max_period}, (int,))
    if x_ticks is not None:
        validate_input_types({"x_ticks": x_ticks}, (list,))
    if figsize is not None:
        validate_input_types({"figsize": figsize}, (tuple,))
    if dataframe[columns].isna().sum().sum() > 0:
        raise ValueError(
            "Input dataframe must not contain missing values for the columns to be analysed!"
        )

    # import required libraries
    import numpy as np
    from statsmodels.formula.api import ols
    from sklearn.feature_selection import mutual_info_regression
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # define plot style
    plt.style.use("fivethirtyeight")

    # define default figsize in case input was none
    if figsize is None:
        figsize = (10, len(columns) * 3)

    # define figure layout
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = GridSpec(nrows=len(columns), ncols=1, figure=fig)
    # define figure title
    fig.suptitle(f"Potential of trigonometric feature engineering\nbased on {'mutual information' if mutual_info else 'linear regression'}")

    # instanciate an empty dict to store trigonometric potentials
    dict_trig = dict()

    # iterate over columns
    for idx, col in enumerate(columns):
        # instanciate empty dict to store power results
        dict_power_coefs = dict()

        # iterate over all periods range
        for period in range(min_period, max_period + 1):
            # make a copy of dataframe
            df_trig = dataframe[col].copy().to_frame()

            # create a time column
            df_trig["time_idx"] = np.arange(len(df_trig), dtype=np.float32)
            # create sine and cosine features for the given period
            df_trig[f"target_sin_period_{period}"] = df_trig["time_idx"].apply(
                lambda x: np.sin(x * (2.0 * np.pi / period))
            )
            df_trig[f"target_cos_period_{period}"] = df_trig["time_idx"].apply(
                lambda x: np.cos(x * (2.0 * np.pi / period))
            )

            # check modelling flag
            if not mutual_info:

                # model a linear regression with sine and cosine features
                model = ols(
                    formula=f"{col} ~ target_sin_period_{period} + target_cos_period_{period}",
                    data=df_trig,
                ).fit()
                # get sine and cosine params from OLS
                sin_param = model.params[f"target_sin_period_{period}"]
                cos_param = model.params[f"target_cos_period_{period}"]
                # calculate overall coeficient for the given period
                power = np.sqrt(sin_param**2 + cos_param**2)

            # mutual information power
            else:
                # calculate mutual information of sin and cossine features
                mi = mutual_info_regression(
                    X=df_trig[[f"target_sin_period_{period}", f"target_cos_period_{period}"]], 
                    y=df_trig[col], 
                    discrete_features='auto', 
                    n_neighbors=3, 
                    copy=True
                )
                # calculate overall coeficient for the given period
                power = np.sqrt(np.sum(mi**2))

            # add power coef to dict with all results
            dict_power_coefs[period] = power

        # create a dataframe from power dict
        df_power = (
            pd.DataFrame.from_dict(
                data=dict_power_coefs, orient="index", columns=["power"]
            )
            .reset_index()
            .rename(columns={"index": "period"})
        )

        # save trigonometric results
        dict_trig[col] = df_power
        
        # plot
        axs = fig.add_subplot(gs[idx, 0])
        axs.step(x=df_power["period"], y=df_power["power"], where="mid")
        # define plot details
        axs.set_title(f"{col}")
        axs.set_ylabel("Estimated power")
        axs.set_xlabel("Period")
        axs.set_ylim(bottom=-0.01)
        if x_ticks is not None:
            axs.set_xticks(x_ticks)
            axs.ticklabel_format(style="plain")

    # display
    plt.show()

    return dict_trig


def inspect_periodograms(
    dataframe: pd.DataFrame,
    columns: List[str],
    sampling_frequency: int,
    x_ticks: Union[list, None] = None,
    figsize: Union[tuple, None] = None,
) -> dict:
    """
    Plot periodogram for the input column on the input dataframe.

    Args
        dataframe: a pd.DataFrame with columns to inspect periodogram.
        columns: a list of columns to inspect periodogram.
        sampling_frequency: an int with sampling frequency as required by scipy.signal.periodogram
        x_ticks: a list with ticks to display on plot in case it makes easier
            to interpret results
        figsize: a tuple with figure size for the whole plot

    Return
        dict_periodogram: a dict where keys are columns and values
            are their respective periodograms results
    """

    # input validation
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"columns": columns}, (list,))
    validate_dataframe_cols(
        dataframe=dataframe, columns=tuple(columns), error_msg="Incorrect column names!"
    )
    validate_input_types({"sampling_frequency": sampling_frequency}, (int, float))
    if x_ticks is not None:
        validate_input_types({"x_ticks": x_ticks}, (list,))
    if figsize is not None:
        validate_input_types({"figsize": figsize}, (tuple,))
    if dataframe[columns].isna().sum().sum() > 0:
        raise ValueError(
            "Input dataframe must not contain missing values for the columns to be analysed!"
        )

    # import required libraries
    import matplotlib.pyplot as plt
    from scipy.signal import periodogram
    from matplotlib.gridspec import GridSpec

    # define plot style
    plt.style.use("fivethirtyeight")

    # define figure layout
    fig = plt.figure(
        figsize=(10, 3*len(columns)) if figsize is None else figsize,
        layout="constrained")
    gs = GridSpec(nrows=len(columns), ncols=1, figure=fig)

    # instanciate an empty dict to store trigonometric potentials
    dict_periodogram = dict()

    # iterate over columns to plot
    for idx, col in enumerate(columns):
        ###########################
        ####### PERIODOGRAM #######
        # https://www.kaggle.com/code/ryanholbrook/seasonality

        # calculate periodogram
        #   frequency: Array of sample frequencies.
        #   spectrum: Power spectral density or power spectrum of x.
        frequencies, power_spectrum = periodogram(
            x=dataframe[col].values,
            fs=sampling_frequency,
            detrend="linear",  # how to detrend each segment
            scaling="spectrum",  # power spectral density -> Pxx has units of V**2
        )

        # create a dataframe with periodogram results
        df_periodogram = pd.DataFrame(
            data={"freq": frequencies, "power": power_spectrum}
        )
        # sort dataframe by power spectrum
        df_periodogram = df_periodogram.sort_values(by="power", ascending=False)

        # save periodogram info back to all periodograms info
        dict_periodogram[col] = df_periodogram

        # plot periodogram
        axs = fig.add_subplot(gs[idx, 0])
        axs.step(frequencies, power_spectrum)
        # define plot details
        axs.set_title(f"\nPeriodogram: {col}")
        axs.set_ylabel("Spectral density")
        axs.set_xlabel("Frequencies")
        if x_ticks is not None:
            axs.set_xticks(x_ticks)

    # display chart
    plt.show()

    return dict_periodogram


def inspect_anomalies(
    dataframe: pd.DataFrame,
    columns: List[str],
    period: int,
    std_threshold: int = 2,
    loess_decomp: bool = True,
    moving_avg_add_params: Union[dict, None] = dict(),
    loess_add_params: Union[dict, None] = dict(),
    figsize: Union[tuple, None] = None,
) -> None:
    """ "
    Compare moving LOESS (or moving average) seasonal decomposition over diferent features
    given the input params.

    Args
        dataframe: a pd.DataFrame with columns to plot anamalies
        columns: a list with the name of columns to plot anomaly analysis
        period: an int with the period param for LOESS decomposition
        std_threshold: an int with how many std from the mean an error
            is considered as an anomaly
        loess_decomp: a bool to indicate of use LOESS decomposition [loess_decomp = True] or
            moving average decomposition [loess_decomp = False]
        moving_avg_add_params: a dict with params for statsmodels.tsa.seasonal.seasonal_decompose function
        loess_add_params: a dict with params for statsmodels.tsa.seasonal.STL function
        figsize: a tuple with figure size of plot for each column

    Return
        None
    """

    # input validation
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"columns": columns}, (list,))
    validate_dataframe_cols(
        dataframe=dataframe, columns=tuple(columns), error_msg="Incorrect column names!"
    )
    validate_input_types({"period": period}, (int,))
    validate_input_types({"std_threshold": std_threshold}, (int,))
    if figsize is not None:
        validate_input_types({"figsize": figsize}, (tuple,))
    validate_input_types({"loess_decomp": loess_decomp}, (bool,))
    validate_input_types({"moving_avg_add_params": moving_avg_add_params}, (dict,))
    validate_input_types({"loess_add_params": loess_add_params}, (dict,))
    if "x" in moving_avg_add_params.keys():
        raise ValueError(
            "'x' param must not be input on moving_avg_add_params. Input it by means of the 'dataframe' param."
        )
    if "period" in moving_avg_add_params.keys():
        raise ValueError(
            "'period' param must not be input on moving_avg_add_params. Input it by means of the 'period' param."
        )
    if "endog" in loess_add_params.keys():
        raise ValueError(
            "'endog' param must not be input on loess_add_params. Input it by means of the 'dataframe' param."
        )
    if "period" in loess_add_params.keys():
        raise ValueError(
            "'period' param must not be input on loess_add_params. Input it by means of the 'period' param."
        )
    if figsize is not None:
        validate_input_types({"figsize": figsize}, (tuple,))

    # import required libraries
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
    from statsmodels.tsa.seasonal import seasonal_decompose, STL

    # define plot style
    plt.style.use("fivethirtyeight")

    # define figure layout
    fig = plt.figure(
        figsize= (10, len(columns) * 3) if figsize is None else figsize, 
        layout="constrained")
    gs = GridSpec(nrows=len(columns), ncols=1, figure=fig)

    # iterate over columns to plot
    for idx, col in enumerate(columns):

        ######################################
        ####### SEASONAL DECOMPOSITION #######
        # check if decomposition is not LOESS
        if not loess_decomp:
            # define decomposition params for moving avg decomposition
            moving_avg_default_params = dict(
                x=dataframe[col],  # user input
                period=period,  # user input
                model="multiplicative",
                filt=None,
                two_sided=True,
                extrapolate_trend="freq",
            )
            # overwrite decomposition params with the inputed ones
            moving_avg_params = {
                k: (
                    v
                    if moving_avg_add_params.get(k) is None
                    else moving_avg_add_params[k]
                )
                for k, v in moving_avg_default_params.items()
            }

            # perform seasonal decomposition using moving averages
            sd = seasonal_decompose(**moving_avg_params)

        # LOESS decomposition
        else:
            # define decomposition params for moving avg decomposition
            loess_default_params = dict(
                endog=dataframe[col],  # user input
                period=period,  # user input
                seasonal=7,
                trend=None,
                low_pass=None,
                seasonal_deg=1,
                trend_deg=1,
                low_pass_deg=1,
                robust=False,
                seasonal_jump=1,
                trend_jump=1,
                low_pass_jump=1,
            )
            # overwrite decomposition params with the inputed ones
            loess_params = {
                k: (v if loess_add_params.get(k) is None else loess_add_params[k])
                for k, v in loess_default_params.items()
            }

            # perform seasonal decomposition using LOESS.
            sd = STL(**loess_params).fit()

        #################################
        ####### ANOMALY DETECTION #######
        # define residual variable
        resid = sd.resid

        # get residual mean and std
        resid_mu = resid.mean()
        resid_dev = resid.std(ddof=1)

        # define confidence intervals
        lower_ci = resid_mu - std_threshold * resid_dev
        upper_ci = resid_mu + std_threshold * resid_dev

        # convert pandas series to pandas dataframe
        df_resid = resid.to_frame(name="resid")
        # assign anomaly flag according to confidence interval of errors
        df_resid["anomaly"] = df_resid["resid"].apply(
            lambda x: True if (x < lower_ci) or (x > upper_ci) else False
        )

        # merge anomaly dataframe to seasonal dataframe
        df_anomaly = pd.merge(
            left=df_resid,
            right=dataframe[col],
            left_index=True,
            right_index=True,
            how="left",
        ).reset_index(names=["timestamp"])

        # plot original data and anomaly flag
        axs = fig.add_subplot(gs[idx, 0])
        sns.lineplot(
            data=df_anomaly,
            x="timestamp",
            y=col,
            color="#0070b2",
            alpha=0.4,
            ax=axs,
        )
        sns.scatterplot(
            data=df_anomaly[df_anomaly["anomaly"]],
            x="timestamp",
            y=col,
            c="#ef4e23",
            marker="X",
            edgecolors="black",
            ax=axs,
        )
        # define plot details
        axs.set_title(
            f'{col}:\n{"moving average" if not loess_decomp else "LOESS"} anomaly detection'
            )
        axs.set_ylabel(col)
        axs.set_xlabel("Timestamps")
        axs.legend(
            handles=[
                mpatches.Patch(color="#0070b2", alpha=0.7, label="Normal data"),
                mpatches.Patch(color="#ef4e23", label="Anomaly"),
            ],
            bbox_to_anchor=(1.12, 1),
            title="Data point",
        )

    # display
    plt.show()


def granger_test(
    dataframe: pd.DataFrame, 
    target_col: str,
    feature_cols: list,
    maxlags: int, 
    addconst: bool=True, 
    report: bool=True
) -> dict:
    """
    Perform granger causality test: whether the time series in the second column 
    Granger causes the time series in the first column.  
    Performed test is based on statsmodels.tsa.stattools.grangercausalitytests.

    Args
        dataframe: a pd.DataFrame with columns
        target_col: a str with the name of the target column
        feature_cols: a list with the name of the features that may be good predictors
            of the target col.
        maxlags: an integer indicating up to what lag test must be performed.
        addconstbool: a bool to indicate if has to include a constant in the model.
        report: a bool flag to indicate if report has to be displayed
        
    Return
        df_summary: a pd.DataFrame with granger causal test for each feature. 
            Please read note bellow for further details.

        
    NOTES:
        
        GRANGER-CAUSALITY TEST
        
        The Null hypothesis:
            The time series in the second column [feature] does NOT Granger cause the time series in the first column [target]

        - Grange causality means that past values of the second column [feature] have a statistically significant effect on the current value of the first column [target], 
            taking past values of the first column into account as regressors.
        - The null hypothesis for the F-test is that the coefficients corresponding to past values of the second time series [feature] are zero.
        - We reject the null hypothesis that the second column [feature]does not Granger cause the first column [target] if the p-values are below a desired size of the test.            
    """

    # input validation
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"target_col": target_col}, (str,))
    validate_input_types({"feature_col": feature_cols}, (list,))
    validate_dataframe_cols(
        dataframe=dataframe, columns=tuple([target_col, *feature_cols]), error_msg="Incorrect column names!"
    )
    validate_input_types({"maxlags": maxlags}, (int,))
    validate_input_types({"addconst": addconst}, (bool,))    
    validate_input_types({"report": report}, (bool,))

    # import required libraries
    from statsmodels.tsa.stattools import grangercausalitytests
    import warnings

    # instanciate an empty dict to save results
    df_summary = pd.DataFrame()

    # iterate over each col on feature col:
    for feature_col in feature_cols:
        # open warning manager
        with warnings.catch_warnings():
            # supress verbose param warning
            warnings.simplefilter(action='ignore', category=FutureWarning)
            # perform granger test
            g_test = grangercausalitytests(
                x=dataframe[[target_col, feature_col]], 
                # If an integer, computes the test for all lags up to maxlag. 
                # If an iterable, computes the tests only for the lags in maxlag.
                maxlag=maxlags, 
                addconst=addconst, # Include a constant in the model.
                verbose=False # don't print results
            )

        # prepare a results dict with lags and F-test p-value
        dict_summary = {
            f"p_value up to lag {k}": v[0]["params_ftest"][1] 
            for k, v 
            in g_test.items()
            }
        # create a column to indicate feature
        dict_summary["feature"] = feature_col

        # convert dict to dataframe        
        df_summary_col = pd.DataFrame(data=dict_summary, index=[0])

        # concatenate col summary with all summaries
        df_summary = pd.concat(objs=[df_summary, df_summary_col], axis=0, ignore_index=True)

    # get name of all columns with p-values
    p_cols = [col for col in df_summary.columns if col.startswith("p_value")]
    # create a column to indicate the minimum p-value over all lags
    df_summary["min p-value over all lags"] = df_summary[p_cols].min(axis=1)
    # reorder dataframe columns
    df_summary = df_summary[
        ["feature", "min p-value over all lags"] + p_cols
    ]
    # sort dataframe by min p-value
    df_summary = df_summary.sort_values(by="min p-value over all lags", ascending=True)

    # check if user wants report
    if report:
        # print report
        print(
            "GRANGER-CAUSALITY TEST\n\n"
            "The Null hypothesis:\n"
            "\tThe time series in the second column [feature] does NOT Granger cause the time series in the first column [target].\n\n"
            "NOTES:\n"
            "\tGrange causality means that past values of the second column [feature] have a statistically significant effect on the current value of the first column [target], \n" 
            "\t\ttaking past values of the first column into account as regressors.\n"
            "\tThe null hypothesis for the F-test is that the coefficients corresponding to past values of the second time series [feature] are zero.\n"
            "\tWe reject the null hypothesis that the second column [feature] does not Granger cause the first column [target] if the p-values are below a desired size of the test.\n"
        )
        display(df_summary)

    return df_summary


def inspect_seasonalities(
    dataframe: pd.DataFrame,
    date_col: str, 
    target_cols: list,
    boxplot: bool=True,
    show_outliers: bool=False
) -> pd.DataFrame:
    """
    Create time-related features to explore possible seasonality
    of datetime column (date_col) and a terget column (target_col).

    Args
        dataframe: a pd.DataFrame with date_col and target_col to inspect.
        date_col: a str with the name of the timestamp column.
        target_cols: a list with the names of the target columns to inspect
            possible seasonalities.
        boxplot: a bool to indicate if plot boxplot for features with few unique values,
            in case boxplot = True, or plot lineplot for all features, in case boxplot = False.
        show_outliers: a bool flag to indicate if 
            outliers are to be displayed on boxplot.
            
    Return
        dict_seasonalities: a dict where keys are target columns and values are pd.DataFrames
            with the input dataframe and features as well as the created time related features for the analysis. 
            Check note below to undestand created features.

    
    Note: 
        The created features to inspect seasonality are the following ones:
            year - Year with century as a decimal number.
            month of year - Month as a zero-padded decimal number.
            week of month - Week of the month
            week of year - Week number of the year (Monday as the first day of the week) as a zero-padded decimal number. All days in a new year preceding the first Monday are considered to be in week 0.
            day of week - Weekday as a decimal number, where 0 is Sunday and 6 is Saturday.
            day of month - Day of the month as a zero-padded decimal number.
            day of year - Day of the year as a zero-padded decimal number.
    """

    # input validation
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"date_col": date_col}, (str,))
    validate_input_types({"target_cols": target_cols}, (list,))
    validate_dataframe_cols(
        dataframe=dataframe, columns=tuple([date_col, *target_cols]), error_msg="Incorrect column names!"
    )
    validate_input_types({"boxplot": boxplot}, (bool,))
    validate_input_types({"show_outliers": show_outliers}, (bool,))

    # import required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # define plot style
    plt.style.use("fivethirtyeight")

    # create an empty dict to store seasonalities per target
    dict_seasonalities = dict()

    # define a dict to map titles
    dict_cols_to_plot = dict(
        y="year", 
        moy="month of year", 
        wom="week of month", 
        woy="week of year", 
        dow="day of week", 
        dom="day of month", 
        doy="day of year"
    )

    # create a dict with diferent colors to plot
    dict_color = {
        "y": "#61005d", # purple
        "moy": "#0070b2", # blue
        "wom": "#6c6c6c", # green
        "woy": "#c38d10", # green
        "dow": "#d72f0b", # red
        "dom": "#d72f0b", # red
        "doy": "#d72f0b", # red
    }

    # iterate over target columns
    for target_col in target_cols:

        # make a copy of input dataframe
        df = dataframe[[date_col, target_col]].copy()

        # check if date_col is a datetype column
        if date_col not in dataframe.select_dtypes(include=[np.datetime64]).columns:
            # raise error
            raise ValueError(f"date_col column must be a column with dtype datetime64")

        # get year - %Y - year with century as a decimal number
        df["y"] = df[date_col].dt.strftime("%Y").astype(int)
        # get month of year - %m - Month as a zero-padded decimal number.
        df["moy"] = df[date_col].dt.strftime("%m").astype(int)
        # week of month
        df["wom"] = ( # day of month - 1 // 7 + 1
            (df[date_col].dt.strftime("%d").astype(int) - 1) // 7 + 1
            )
        # week of year - %W - Week number of the year (Monday as the first day of the week) as a zero-padded decimal number. 
        # All days in a new year preceding the first Monday are considered to be in week 0.
        df["woy"] = df[date_col].dt.strftime("%W").astype(int)
        # day of week - %w - Weekday as a decimal number, where 0 is Sunday and 6 is Saturday.
        df["dow"] = df[date_col].dt.strftime("%w").astype(int)
        # day of month - %d - Day of the month as a zero-padded decimal number.
        df["dom"] = df[date_col].dt.strftime("%d").astype(int)
        # day of year - %j - Day of the year as a zero-padded decimal number.
        df["doy"] = df[date_col].dt.strftime("%j").astype(int)

        # assign target seasonalities to all targets seasonalities
        dict_seasonalities[target_col] = df

        # define plot
        f, axs = plt.subplots(
            nrows=len(dict_cols_to_plot.keys()),
            ncols=1,
            figsize=(10, len(dict_cols_to_plot.keys()) * 3),
            constrained_layout=True,
        )
        # define figure title
        f.suptitle(target_col.upper())

        # iterate over columns to plot
        for idx, col in enumerate(dict_cols_to_plot.keys()):
            # check how many unique values on column
            if (df[col].nunique() < 15) and (boxplot):
                # plot a box chart
                sns.boxplot(
                    data=df, x=col, y=target_col,
                    orient="v",
                    meanline=True, showmeans=True, meanprops={"color": "black", "marker": "*"},
                    showfliers=show_outliers,
                    ax=axs[idx]
                    )
            # high number of unique values
            else:
                # plot a line chart
                sns.lineplot(
                    data=df, x=col, y=target_col, 
                    #color="#d72f0b", 
                    color=dict_color[col],
                    estimator=np.nanmean,
                    errorbar=("ci", 95), n_boot=1_000,
                    err_kws={"alpha": .4},
                    err_style="band",
                    marker="*", markerfacecolor="black", markeredgecolor="black", markersize=3,
                    ax=axs[idx]
                    )

            # define plot details
            axs[idx].set_title(f"\n{target_col} \nvs {dict_cols_to_plot[col]}")
            axs[idx].set_ylabel(target_col)
            axs[idx].set_xlabel(dict_cols_to_plot[col])
            if col in ["y", "wom"]:
                unique_x = df[col].unique()
                unique_x.sort()
                axs[idx].set_xticks(unique_x)
            if (axs[idx].get_ylim()[0] < 0 < axs[idx].get_ylim()[1]):
                axs[idx].axhline(y=0.0, xmin=0.01, xmax=0.99, color="black", alpha=1, linestyle=":", linewidth=1)
            else:
                axs[idx].set_ylim(ymin=0)

        # display 
        plt.show()

    return dict_seasonalities


def calculate_performance(
    y_true: np.array,
    y_pred: np.array,
    horizons: list,
    benchmark: Union[float, int] = None
) -> pd.DataFrame:
    """
    Calculate time series metrics over different horizons given the inputs.
    Calculated metrics are: 
        MPE = Mean Percentage Error
        MAE = Mean Absolute Error
        Relative MAE = MAE(model) / MAE(benchmark)
        MAPE = Mean Absolute Percentage Error
        Maximum APE (Absolute Percentage Error)
        sMAPE = Symmetric mean absolute percentage error
        RMSE = Mean Squared Error
        SE = Sum Errors = sum(errors) = sum( real - pred )
        PSE = Percent Sum Error = sum(errors) / sum(real) = sum( real - pred ) / sum(real)

        
    Args
        y_true: a np.array with real values
        y_pred: a np.array with predicted values
        horizons: a list with horizons to calculate metrics
        benchmark: a float or int with benchmarch prediction for all horizonst    

    Return
        df_metrics: a pd.DataFrame with metrics over all horizons.

    """

    # import libraries
    from project_lib.input_validation import validate_input_types
    import numpy as np
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    )

    # input validation
    validate_input_types({"y_true": y_true}, (np.ndarray,))
    validate_input_types({"y_pred": y_pred}, (np.ndarray,))
    validate_input_types({"horizons": horizons}, (list,))
    if benchmark is not None:
        validate_input_types({"benchmark": benchmark}, (int, float))

    # instanciate an empy dict to create dataframe later
    dict_metrics = dict()

    # iterate over horizons
    for horizon in horizons:
        # filter input based on horizons
        y_true_horizon, y_pred_horizon = y_true[:horizon], y_pred[:horizon]

        # sanity check
        assert len(y_true_horizon) == len(y_pred_horizon), ( # compare to horizon ???
            f"Error on horizon filtering! len(y_true_horizon) = {len(y_true_horizon)}, len(y_pred_horizon) = {len(y_pred_horizon)}"
            )

        # calculate metrics
        # MPE = Mean Percentage Error
        dict_metrics[f"MPE_D+{horizon}"] = np.mean(             # mean of
            (y_true_horizon - y_pred_horizon) / y_true_horizon  # percentage error
            )
        # MAE = Mean Absolute Error
        dict_metrics[f"MAE_D+{horizon}"] = mean_absolute_error( 
            y_true=y_true_horizon,
            y_pred=y_pred_horizon
            )
        # relative MAE:
        # https://www.statworx.com/en/content-hub/blog/what-the-mape-is-falsely-blamed-for-its-true-weaknesses-and-better-alternatives/
        if benchmark is not None:
            # define bench according to input
            benchmark_pred = np.full(shape=len(y_true_horizon), fill_value=benchmark)
            # calculate focal and benchmark MAE
            focal_mae = mean_absolute_error( 
            y_true=y_true_horizon,
            y_pred=y_pred_horizon
            )
            benchmark_mae = mean_absolute_error( 
            y_true=y_true_horizon,
            y_pred=benchmark_pred
            )
            # relative MAE = MAE(model) / MAE(benchmark)
            dict_metrics[f"relative_MAE_D+{horizon}"] = focal_mae / benchmark_mae
        # MAPE = Mean Absolute Percentage Error
        dict_metrics[f"MAPE_D+{horizon}"] = mean_absolute_percentage_error(
                y_true=y_true_horizon,
                y_pred=y_pred_horizon
            )
        # Maximum APE (Absolute Percentage Error)
        dict_metrics[f"Max_APE_D+{horizon}"] = np.max(                   # max of
            np.abs( (y_true_horizon - y_pred_horizon) / y_true_horizon ) # absolute percentage error
            )
        # sMAPE = Symmetric mean absolute percentage error
        # https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
        dict_metrics[f"sMAPE_D+{horizon}"] = (
            100 * 1/len(y_true_horizon) * np.sum( 2 * np.abs(y_pred_horizon - y_true_horizon) / (np.abs(y_pred_horizon) + np.abs(y_true_horizon)))
            )
        # RMSE = Mean Squared Error
        dict_metrics[f"RMSE_D+{horizon}"] = mean_squared_error( 
            y_true=y_true_horizon,
            y_pred=y_pred_horizon,
            squared=False # returns RMSE value
            )         
        # SE = Sum Errors = sum(errors) = sum( real - pred )
        dict_metrics[f"Sum_Errors_D+{horizon}"] = np.sum(
            (y_true_horizon - y_pred_horizon) # error
            )
        # PSE = Percent Sum Error = sum(errors) / sum(real) = sum( real - pred ) / sum(real)
        dict_metrics[f"Percent_Sum_Errors_D+{horizon}"] = (
            (np.sum(y_true_horizon) - sum(y_pred_horizon)) / np.sum(y_true_horizon)
            )
            
    # create a pandas dict with dict metrics
    df_metrics = pd.DataFrame(data=dict_metrics, index=[0])

    return df_metrics