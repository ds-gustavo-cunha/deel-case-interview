##############################
####### INITIAL CONFIG #######
##############################

# import required library to configure module
import pandas as pd
from project_lib.initial_config import initial_settings
from project_lib.input_validation import validate_input_types

# set the basic cofiguration for this module
initial_settings()

################################
####### MODULE FUNCTIONS #######
################################


def numerical_plot(
    dataframe: pd.DataFrame,
    n_cols: int = 3,
    figsize: tuple = None,
    hist: bool = True,
    save_fig: str = None,
) -> None:
    """
    Plot histogram (or kde) and boxplot for every numerical column on the dataframe.

    Args
        dataframe: a pandas datataframe with numerical features.
        n_cols:: an integer with the number of column in the figure template.
        figsize: a tuple with figsize (width, height) in inches.
        hist: boolean to indicate if user wants a histplot or a kdeplot.
            This may be useful when histplot is too slow.
        save_fig: a string with the path to save the figure

    Return
        None: a None Type object
    """

    # import required libraries
    import seaborn as sns
    import matplotlib.pyplot as plt
    from math import ceil
    from matplotlib import gridspec

    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"n_cols": n_cols}, (int,))
    validate_input_types({"hist": hist}, (bool,))
    if figsize is not None:
        validate_input_types({"figsize": figsize}, (tuple,))
    if save_fig is not None:
        validate_input_types({"save_fig": save_fig}, (str,))

    # display information for user
    print(
        f"In case the function is taking too much time to plot, you can try: hist = False [default: hist=True]."
    )

    # get numeric variables
    df_num = dataframe.select_dtypes(include=["number"])

    # define number of rows given the required number of columns (n_cols)
    n_rows = 2 * (ceil((df_num.shape[1] - 1) // n_cols) + 1)

    # check if user input figsize
    if figsize is None:
        # assign th default figsize
        figsize = (n_cols * 6, n_rows * 4.5)

    # create a figure object
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    # create grid for plotting
    specs = gridspec.GridSpec(ncols=n_cols, nrows=n_rows, figure=fig)

    # iterate over columns to be plotted
    for index, column in enumerate(df_num.columns):
        # define row index for histogram and box plot
        hist_index = (index // (n_cols)) * 2
        box_index = hist_index + 1

        # create subplots to plot the given feature
        hist_axs = fig.add_subplot(specs[hist_index, index % n_cols])  # row, column
        box_axs = fig.add_subplot(specs[box_index, index % n_cols], sharex=hist_axs)  # row, column

        # check if user wants histplot
        if hist:
            # set title for hist plot
            hist_axs.set_title(column.upper())
            # plot histogram
            sns.histplot(x=column, data=df_num, ax=hist_axs, kde=True)

        # in case user want kdeplot instead of histplot
        else:
            # set title for kde plot
            hist_axs.set_title(column.upper())
            # plot kdeplot
            sns.kdeplot(x=column, data=df_num, ax=hist_axs, fill=True)

        # set title for box plot
        box_axs.set_title(column.upper())
        # plot boxplot
        sns.boxplot(x=column, data=df_num, ax=box_axs)

    # check if user wants to save figure
    if save_fig is not None:
        # save figure
        plt.savefig(save_fig, facecolor="white", bbox_inches="tight")

    return None


def categorical_plot(
    dataframe: pd.DataFrame,
    max_num_cat: int = 10,
    n_cols: int = 3,
    trunc_label: int = 20,
    figsize: bool = None,
    save_fig: str = None,
) -> None:
    """
    Plot horizontal bar charts for all categorical features in the dataframe,
    showing count and percentage labels on each bar.

    Args
        dataframe: a pandas datataframe with categorical features.
        max_num_cat: an integer with the maximum number of categories to be displayed on bar plot.
        n_cols: is a integer with the number of columns on the final chart.
        trunc_label: a integer with the maximum number of characters for the y label ticks.
        figsize: tuple with figsize (width, height) in inches.
        save_fig: a string with the path to save the figure

    Return
        None: a NoneType object

    NOTE:
        Names of columns on the y axis are truncated (to "trunc_label" characters).
        When a column has more categories than max_num_cat, the top (max_num_cat - 1)
        are shown individually and the rest are grouped as "OTHER (N categories)".
    """

    # import required libraries
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import seaborn as sns

    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"max_num_cat": max_num_cat}, (int,))
    validate_input_types({"n_cols": n_cols}, (int,))
    validate_input_types({"trunc_label": trunc_label}, (int,))
    if figsize is not None:
        validate_input_types({"figsize": figsize}, (tuple,))
    if save_fig is not None:
        validate_input_types({"save_fig": save_fig}, (str,))

    # get string variables
    df_cat = dataframe.select_dtypes(include=["object"])

    # define number of rows
    n_rows = df_cat.shape[1] // n_cols + 1

    # check if user input figsize
    if figsize is None:
        figsize = (n_cols * 6, n_rows * 4.5)

    # create a figure object
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    # create grid for plotting
    specs = gridspec.GridSpec(ncols=n_cols, nrows=n_rows, figure=fig)

    total_rows = len(dataframe)

    # iterate over column to plot countplot figure
    for index, column in enumerate(df_cat.columns):
        # create a subplot to plot the given feature
        ax1 = fig.add_subplot(specs[index // n_cols, index % n_cols])

        # count values per category (sorted descending by frequency)
        count_series = df_cat[column].value_counts()
        n_unique = len(count_series)
        # make sure the indexes are strings
        count_series.index = [str(i) for i in count_series.index.tolist()]

        # check if number of categories is greater than threshold
        if n_unique > max_num_cat:
            n_remaining = n_unique - (max_num_cat - 1)
            count_after_threshold = count_series[max_num_cat - 1:].sum()
            plot_series = count_series[: max_num_cat - 1].copy()
            plot_series[f"OTHER ({n_remaining} categories)"] = int(count_after_threshold)
            title_suffix = f" [top {max_num_cat - 1} of {n_unique}]"
        else:
            plot_series = count_series.copy()
            title_suffix = f" [{n_unique} categories]"

        # truncate long labels
        tick_labels = [str(label)[:trunc_label] for label in plot_series.index.tolist()]

        # plot horizontal bar chart
        bars = ax1.barh(
            y=range(len(plot_series)),
            width=plot_series.values,
            color=sns.color_palette("rainbow", len(plot_series)),
        )

        # set y-tick positions and labels
        ax1.set_yticks(range(len(plot_series)))
        ax1.set_yticklabels(tick_labels, fontsize=9)

        # invert y-axis so highest-frequency bar is at the top
        ax1.invert_yaxis()

        # add count and percentage labels on bars
        x_max = plot_series.values.max()
        for bar, count in zip(bars, plot_series.values):
            pct = count / total_rows * 100
            label = f" {count:,} ({pct:.1f}%)"
            ax1.text(
                bar.get_width() + x_max * 0.01,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left",
                fontsize=8,
            )

        # expand x-axis to make room for labels
        ax1.set_xlim(0, x_max * 1.35)

        # set title and axis labels
        ax1.set_title(f"{column.upper()}{title_suffix}", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Count")
        ax1.set_ylabel("")

    # check if user wants to save figure
    if save_fig is not None:
        plt.savefig(save_fig, facecolor="white", bbox_inches="tight")

    return None  # explicitly


def datetime_plot(
    dataframe: pd.DataFrame, n_cols: int = 3, figsize: bool = None, save_fig: str = None
) -> None:
    """
    Plot histogram for all datetime features in the dataframe.
    Dataframe is supposed to have only datetime columns.

    Args
        df_cat: a pandas datataframe with datetime columns.
        n_cols: is a integer with the number of columns on the final chart.
        figsize: tuple with figsize (width, height) in inches.
        save_fig: a string with the path to save the figure

    Return
        None: a NoneType object
    """

    # import required libraries
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import seaborn as sns

    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"n_cols": n_cols}, (int,))
    if figsize is not None:
        validate_input_types({"figsize": figsize}, (tuple,))
    if save_fig is not None:
        validate_input_types({"save_fig": save_fig}, (str,))

    # get string variables
    df_date = dataframe.select_dtypes(include=["datetime", "datetime64[ns, UTC]"])

    # define number of rows
    n_rows = df_date.shape[1] // n_cols + 1

    # check if user input figsize
    if figsize is None:
        # assign th default figsize
        # figsize = (n_cols*4.5, n_rows*4.5)
        figsize = (n_cols * 6, n_rows * 4.5)

    # create a figure object
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    # create grid for plotting
    specs = gridspec.GridSpec(ncols=n_cols, nrows=n_rows, figure=fig)

    # iterate over column to plot countplot figure
    for index, column in enumerate(df_date.columns):
        # create a subplot to plot the given feature
        ax1 = fig.add_subplot(specs[index // n_cols, index % n_cols])
        # set the title for the subplot
        ax1.set_title(column.upper())
        # plot histogram
        sns.histplot(x=column, data=df_date, ax=ax1, kde=True)
        # rotate x label to make it easier to read
        plt.xticks(rotation=60)

    # check if user wants to save figure
    if save_fig is not None:
        # save figure
        plt.savefig(save_fig, facecolor="white", bbox_inches="tight")

    return None  # explicitly


def value_counts_proportion(
    dataframe_all: pd.DataFrame,
    dataframe_filter: pd.DataFrame,
    column: str,
    display_results: bool = True,
) -> pd.DataFrame:
    """
    Calculate the value counts proportion for a given column
    of a filtered dataframe in regard to unfiltered dataframe

    Args
        dataframe_all: a pandas dataframe with filtered data
        dataframe_filter: a pandas dataframe with the dataframe
            with filtered data in order to calculate proportion
            in regard to unfiltered data
        column: a string with the name of the column to calculate proportion
        display_results: a boolean to indicate if user wants to display the dataframe (True)
            or only return it (False)

    Return
        df_merged: a pandas dataframe with value counts proportion
            for the chosen column.
    """

    # input verification
    validate_input_types({"dataframe_all": dataframe_all}, (pd.core.frame.DataFrame,))
    validate_input_types(
        {"dataframe_filter": dataframe_filter}, (pd.core.frame.DataFrame,)
    )
    validate_input_types({"column": column}, (str,))

    # get value counts for the dataframe with all data
    df_value_counts_all = pd.DataFrame(dataframe_all[column].value_counts(dropna=False))
    # reset index
    df_value_counts_all.reset_index(inplace=True)
    # rename columns
    df_value_counts_all.rename(
        columns={"index": "label", column: f"{column}_all"}, inplace=True
    )

    # get value counts for the dataframe with filtered data
    df_value_counts_filter = pd.DataFrame(
        dataframe_filter[column].value_counts(dropna=False)
    )
    # reset index
    df_value_counts_filter.reset_index(inplace=True)
    # rename columns
    df_value_counts_filter.rename(
        columns={"index": "label", column: f"{column}_filter"}, inplace=True
    )

    # merge dataframes on filter label keep all labels of dataframe_all [left join]
    df_merged = pd.merge(
        left=df_value_counts_all, right=df_value_counts_filter, on="label", how="left"
    )

    # calculate proportion
    df_merged["percent_proportion_filter_over_all"] = (
        df_merged[f"{column}_filter"] / df_merged[f"{column}_all"]
    ) * 100

    # sort by percent_proportion_filter_over_all
    df_merged = df_merged.sort_values(
        by="percent_proportion_filter_over_all", ascending=False, ignore_index=True
    )

    # check if user wants results to be displayed
    if display_results:
        # display value counts
        display(df_merged)

    return df_merged


def cramer_v_corrected_stat(
    series_one: pd.Series, series_two: pd.Series
) -> pd.DataFrame:
    """
    Calculate crame v statistics for two categorical series

    Args:
        series_one: first categorical dataframe column
        series_two: second categorical dataframe column

    Return:
        corr_cramer_v: corrected Cramer-V statistic

    NOTE: This implementation doesn't handle missing value (e.g. np.nan). It will raise warnings in this case.
    """
    # import required libraries
    import numpy as np
    from scipy.stats import chi2_contingency

    # validate inputs
    validate_input_types({"series_one": series_one}, (pd.core.series.Series,))
    validate_input_types({"series_two": series_two}, (pd.core.series.Series,))

    # create confusion matrix
    cm = pd.crosstab(series_one, series_two)
    # calculate the sum along all dimensions
    n = cm.sum().sum()
    # calculate number of row and columns of confusion matrix
    r, k = cm.shape

    # calculate chi_squared statistics
    chi2 = chi2_contingency(cm)[0]

    # calculate chi_squared correction
    chi2corr = max(0, chi2 - (k - 1) * (r - 1) / (n - 1))
    # calculate k correction
    kcorr = k - (k - 1) ** 2 / (n - 1)
    # calculate r correction
    rcorr = r - (r - 1) ** 2 / (n - 1)

    # calculate corrected cramer-v (suppress divide-by-zero for degenerate columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_cramer_v = np.sqrt((chi2corr / n) / (min(kcorr - 1, rcorr - 1)))

    return corr_cramer_v


def create_cramer_v_dataframe(
    categ_features_analysis_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a correlation matrix for features on categorical dataframe

    Args:
        categ_features_analysis_dataframe: dataframe with only categorical features

    Return:
        categ_corr_matrix: dataframe with cramer-v for every row-column pair
            in the input dataframe
    """
    # validate inputs
    validate_input_types(
        {"categ_features_analysis_dataframe": categ_features_analysis_dataframe},
        (pd.core.frame.DataFrame,),
    )

    columns = categ_features_analysis_dataframe.columns.tolist()

    # create final dataframe skeleton with float dtype
    df_cramer_v = pd.DataFrame(
        columns=columns,
        index=columns,
        dtype=float,
    )

    # diagonal is always 1.0 (perfect association of a variable with itself)
    for col in columns:
        df_cramer_v.loc[col, col] = 1.0

    # leverage symmetry: cramer-v(X, Y) == cramer-v(Y, X)
    for i, row_col in enumerate(columns):
        for j in range(i + 1, len(columns)):
            col_col = columns[j]
            v = float(
                cramer_v_corrected_stat(
                    categ_features_analysis_dataframe[row_col],
                    categ_features_analysis_dataframe[col_col],
                )
            )
            df_cramer_v.loc[row_col, col_col] = v
            df_cramer_v.loc[col_col, row_col] = v

    return df_cramer_v


def plot_cramer_v_heatmap(
    df_cat: pd.DataFrame, figsize: tuple = None, saving_path: str = None
) -> pd.DataFrame:
    """
    Calculate and plot the correlation matrix of all columns in df_cat
    using corrected cramer-v correlation coefficient.

    Args
        df_cat: a pandas dataframe with categorical columns only.
        figsize: a tuple with the figure size (width, height) in case user wants to save it.
        saving_path: a string with the path to save heatmap in case user wants to save it.

    Return
        df_cramer_v_corr: a pandas dataframe with corrected cramer-v correlation coefficient among features.
    """

    # import required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # validate inputs
    validate_input_types({"df_cat": df_cat}, (pd.core.frame.DataFrame,))
    if figsize is not None:
        validate_input_types({"figsize": figsize}, (tuple,))
    if saving_path is not None:
        validate_input_types({"saving_path": saving_path}, (str,))

    # create a dataframe with cramer-v for every row-column pair
    df_cramer_v_corr = create_cramer_v_dataframe(df_cat)

    # check if user set a figsize
    if figsize is None:
        # define figure size reference
        fig_size_ref = df_cat.shape[1]
        # define a multiplier
        multiplier = 0.75
        # define fig_size variable
        figsize = (multiplier * fig_size_ref, multiplier * fig_size_ref)

    # create figure and ax object
    fig, ax = plt.subplots(figsize=figsize)

    # display heatmap of correlation on figure
    sns.heatmap(
        df_cramer_v_corr,
        annot=True,
        vmin=0,
        vmax=1,
        center=0.5,
        square=True,
        ax=ax,
        cmap=sns.diverging_palette(20, 220, n=256),
    )
    # define figure details
    plt.title("SPEARMAN CORRELATION COEFFICIENT")
    plt.yticks(rotation=0)

    # check if user want to save heatmap
    if saving_path is not None:
        # save figure to inspect outside notebook
        plt.savefig(
            saving_path,
            dpi=200,
            transparent=False,
            bbox_inches="tight",
            facecolor="white",
        )

    return df_cramer_v_corr


def plot_spearman_heatmap(
    df_num: pd.DataFrame, figsize: tuple = None, saving_path: tuple = None
) -> pd.DataFrame:
    """
    Calculate and plot the correlation matrix of all columns in df_num
    using Spearman correlation coefficient (so as to get non-linear relationships).

    Args
        df_num: a pandas dataframe with numerical columns only.
        figsize: a tuple with the figure size (width, height) in case user wants to save it.
        saving_path: a string with the path to save heatmap in case user wants to save it.

    Return
        df_spearman_corr: a pandas dataframe with spearman correlation coefficient among features.
    """

    # import required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # validate inputs
    validate_input_types({"df_num": df_num}, (pd.core.frame.DataFrame,))
    if figsize is not None:
        validate_input_types({"figsize": figsize}, (tuple,))
    if saving_path is not None:
        validate_input_types({"saving_path": saving_path}, (str,))

    # checking df_num type
    assert type(df_num) == pd.DataFrame, "df_num must be a pandas dataframe!"

    # calculate spearman correlation for numerical features
    df_spearman_corr = df_num.corr(method="spearman")  # get non-linear relationships

    # check if user set a figsize
    if figsize is None:
        # define figure size reference
        fig_size_ref = df_num.shape[1]
        # define a multiplier
        multiplier = 0.75
        # define fig_size variable
        figsize = (multiplier * fig_size_ref, multiplier * fig_size_ref)

    # create figure and ax object
    fig, ax = plt.subplots(figsize=figsize)

    # display heatmap of correlation on figure
    sns.heatmap(
        df_spearman_corr,
        annot=True,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        ax=ax,
        cmap=sns.diverging_palette(20, 220, n=256),
    )
    # define figure details
    plt.title("SPEARMAN CORRELATION COEFFICIENT")
    plt.yticks(rotation=0)

    # check if user want to save heatmap
    if saving_path is not None:
        # save figure to inspect outside notebook
        plt.savefig(
            saving_path,
            dpi=200,
            transparent=False,
            bbox_inches="tight",
            facecolor="white",
        )

    return df_spearman_corr


def time_weighted_average(
    sequence: list,
    max_number_weights: int,
) -> float:
    """
    Calculate a weighted average taking into account a given sequence and
    the maximmum number of items to consider into average.


    It get, from end to start, as many values as in max_number_weights and sequence permits:
    (1) if max_number_weights > len(sequence) -> take all values in sequence;
    (2) if max_number_weights <= len(sequence) -> take max_number_weights values from end to start.

    Weights follows a linear increasing weights sequence.

    NaN values in input sequence were not considered on weighted average but
    their positions on input array were taken into account when defining weights. Ex.:
    a = np.array([np.nan, 1, np.nan, np.nan, 2])
    ---> weights = [1, 2, 3, 4, 5]
    ---> final weights used for avg calculations = [np.nan, 2, np.nan, np.nan, 5]
    ---> weighted average = ( 1 * 2 + 2 * 5 ) / ( 2 + 5 )

    Args
        sequence: a numpy array input
        max_number_weights: an integer with the number of values to take into consideration

    Return
        weighted_avg: a float with the weighted average
    """

    # import required libraries
    import numpy as np

    # validate inputs
    validate_input_types({"sequence": sequence}, (np.ndarray,))
    validate_input_types({"max_number_weights": max_number_weights}, (int,))

    # get, from end to start, as many values as in max_number_weights and sequence permits
    # if max_number_weights > len(sequence) -> take all values in sequence
    # if max_number_weights <= len(sequence) -> take max_number_weights values from end to start
    array = sequence[-max_number_weights:]

    # create a array to map NaN values in input array
    nan_array = np.array([np.nan] * max_number_weights)

    # start a linear decreasing weights sequence
    # decreasing = more weights to last value than to previous ones
    weights = np.arange(1, len(array) + 1)

    # set weights for NaN values (in input list) to NaN
    weights = np.where(np.isnan(array), np.nan, weights)

    # calculate weighted average
    # taking into account only non-NaN values in input list
    weighted_avg = np.nansum(array * weights) / np.nansum(weights)

    return weighted_avg
