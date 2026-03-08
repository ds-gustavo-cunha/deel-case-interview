##############################
####### INITIAL CONFIG #######
##############################

# import required library to configure module
import re
import pandas as pd
from typing import Union
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


def numeric_summary(dataframe: pd.DataFrame) -> None:
    """
    It displays statistics for numerical features of the dataframe.
    Displayed statistics are: mean, median, std, min, max, range,
        skew, kurtosis and iqr (interquartile range).

    Args
        dataframe: a pd.DataFrame object that the user wants to check statistics

    Return
        None: a None type object

    NOTE:
        NaNs are ignored for the calculated statistics.
    """

    # import required libraris
    import pandas as pd
    import numpy as np

    #########################################################
    # This function requires that Jinja2 library is installed,
    # but it doesn't need to be imported
    # -> pip(env) install Jinja2
    #########################################################

    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))

    # ======= STATISTICS =======

    # get numeric variables
    df_numeric = dataframe.select_dtypes(include=["number", "bool"])
    bool_cols = df_numeric.select_dtypes(include=["bool"]).columns
    df_numeric = df_numeric.copy()
    df_numeric[bool_cols] = df_numeric[bool_cols].astype(int)

    # central tendency statistics - NaNs are ignored
    mean_stats = pd.DataFrame(df_numeric.apply(np.nanmean)).T
    median_stats = pd.DataFrame(df_numeric.apply(np.nanmedian)).T

    # deviation statistics - NaNs are ignored
    std_stats = pd.DataFrame(df_numeric.apply(np.nanstd)).T
    min_stats = pd.DataFrame(df_numeric.apply(np.nanmin)).T
    max_stats = pd.DataFrame(df_numeric.apply(np.nanmax)).T
    range_stats = pd.DataFrame(
        df_numeric.apply(lambda x: x.max(skipna=True) - x.min(skipna=True))
    ).T
    skew_stats = pd.DataFrame(df_numeric.apply(lambda x: x.skew(skipna=True))).T
    kurtosis_stats = pd.DataFrame(df_numeric.apply(lambda x: x.kurtosis(skipna=True))).T
    iqr_stats = pd.DataFrame(
        df_numeric.apply(
            lambda x: np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)
        )
    ).T

    # concatenate statistics
    df_stats = pd.concat(
        [
            mean_stats,
            median_stats,
            std_stats,
            min_stats,
            max_stats,
            range_stats,
            skew_stats,
            kurtosis_stats,
            iqr_stats,
        ]
    ).T.reset_index()

    # rename columns
    df_stats.columns = [
        "attribute",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "range",
        "skew",
        "kurtosis",
        "iqr",
    ]

    # reorder columns
    df_stats = df_stats[
        [
            "attribute",
            "mean",
            "median",
            "std",
            "iqr",
            "min",
            "max",
            "range",
            "skew",
            "kurtosis",
        ]
    ]

    # print statistics for numerical data
    print("\n\nStatistics for Numerical Variables [NaNs are ignored]:")

    # highlight min and max statistics -> help identify 'non-sense' data
    df_stats = df_stats.style.map(
        lambda x: "background-color: Navy; color: White", subset=["min", "max"]
    )
    # display statatistics
    display(df_stats)

    return None  # explicitly


def check_na_unique_dtypes(
    dataframe: pd.DataFrame, many_columns: bool = False
) -> pd.DataFrame:
    """
    It prints the number of NAs, the percentage of NAs, the number of unique values (include NAs)
    and the data type for each column.

    Args
        dataframe: the pandas dataframe that the user wants to check.
        many_columns: a boolean to solve the pandas truncating rows in case
            dataframe has many features.

    Return
        df_info: a pandas dataframe with informatation o NA, unique values and datatypes
    """

    # import required libraris
    import pandas as pd
    import numpy as np

    #########################################################
    # This function requires that Jinja2 library is installed,
    # but it doesn't need to be imported
    # -> pip install Jinja2
    #########################################################

    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"many_columns": many_columns}, (bool,))

    # set cientific notation for pandas
    pd.set_option("display.float_format", "{:,.1f}".format)

    # ======= MEMORY USAGE INFORMATION =======

    # get memory usage in MB
    df_size_in_memory = dataframe.memory_usage(index=True, deep=True).sum() / (10**6)

    # print info
    print("*" * 49)
    print(f"Dataframe size in memory: {df_size_in_memory:,.3f} MB", "\n")

    # ======= DESCRIPTIVE INFORMATION =======

    # create dictionary with descriptive information
    dict_data = {
        "Num NAs": dataframe.isna().sum(axis=0),
        "Percent NAs": (dataframe.isna().mean(axis=0) * 100).apply(
            lambda x: int(np.ceil(x))
        ),
        "Num unique [include NAs]": dataframe.nunique(dropna=False),
        "Data Type": dataframe.dtypes,
    }

    # define a dataframe from dictionary
    df_info = pd.DataFrame(dict_data)

    # # sort values by dtypes -> easier to read
    # df_info.sort_values(by="Data Type", inplace=True)

    # print indications of next info displayed
    print("-----------------------------")
    print("Dataframe overview:")

    # check if user set dataframe to have many features
    if many_columns:
        # open pandas options with context manager
        with pd.option_context(
            "display.max_rows",
            None,
        ):
            # print descriptive data
            display(
                df_info.style.map(
                    lambda x: "background-color: Navy; color: White",
                    subset=["Percent NAs", "Data Type"],
                )
            )

    # dataframe doesn't have many features
    else:
        # print descriptive data
        display(
            df_info.style.map(
                lambda x: "background-color: Navy; color: White",
                subset=["Percent NAs", "Data Type"],
            )
        )

    # ======= SHAPE INFORMATION =======

    # print dataframe shape
    print("-----------------------------")
    print("\n", f"Dataframe shape is {dataframe.shape}", "\n")

    return df_info


def categorical_summary(
    dataframe: pd.DataFrame,
    nunique_threshold: int = 15,
    unique_name_len_threshold: int = 15,
) -> None:
    """
    For the columns whose type is object, it prints the number of NAs, the percentage of NAs,
    the number of unique values (include NAs) and the data type for each column.
    Besides, the unique values for every columns are printed.

    Args
        dataframe: the pandas dataframe that the user wants to check.
        nunique_threshold: an integer with the maximum number of unique values on a column
            so as to display unique values
        unique_name_len_threshold: an integer with the maximum length of a unique value name
            so as to be displayed. If name is longer than this threshold, name will be truncated.


    Return
        None: a none type object.
    """

    # import required libraris
    import pandas as pd
    import numpy as np

    #########################################################
    # This function requires that Jinja2 library is installed,
    # but it doesn't need to be imported
    # -> pip install Jinja2
    #########################################################

    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"nunique_threshold": nunique_threshold}, (int,))
    validate_input_types(
        {"unique_name_len_threshold": unique_name_len_threshold}, (int,)
    )

    # get numeric variables
    df_cat = dataframe.select_dtypes(include=["object"])

    # create dictionary with descriptive information
    dict_data = {
        "Num NAs": df_cat.isna().sum(axis=0),
        "Percent NAs": (df_cat.isna().mean(axis=0) * 100).apply(
            lambda x: int(np.ceil(x))
        ),
        "Num unique [include NAs]": df_cat.nunique(dropna=False),
        "Data Type": df_cat.dtypes,
    }

    # define a dataframe from dictionary
    df_info = pd.DataFrame(dict_data)

    # introduce the section content
    print("Overview of string columns:")

    # open pandas options with context manager
    with pd.option_context(
        "display.max_rows",
        None,
    ):
        # print descriptive data
        display(
            df_info.style.map(
                lambda x: "background-color: Navy; color: White",
                subset=["Percent NAs"],
            )
        )

    # create separation among sections
    print("-------------------------------------------------", "\n")

    # iterate over columns of df_info
    for col in df_cat.columns:
        # calculate unique values for the given column
        col_unique = df_cat[col].unique().tolist()

        # display a preparation for unique values info
        print(
            f"\033[94m--->\033[0m The unique values for \033[94m\033[1m{col}\033[0m\033[0m column are: [\033[1mvalues are truncated\033[0m]",
            "\n",
        )

        # compare how many unique values are available with the chosen threshold
        if len(col_unique) < nunique_threshold:

            # truncate unique value names given the truncate_threshold param
            col_unique = [
                str(unique)[:unique_name_len_threshold] for unique in col_unique
            ]

            # print unique values
            print(col_unique)  # print unique values
            print("-------------------------------------------------", "\n")

        # number of unique values is greater than the chosen threshold
        else:
            # display message
            print(
                f"Column {col} has more than {nunique_threshold} unique values. To avoid a noisy display, they weren't printed. You can change the nunique_threshold param if you do want to print them."
            )
            print("-------------------------------------------------", "\n")

    return None  # explicitly


def check_dataframe(
    dataframe: pd.DataFrame, summary_stats=False, head: bool = False, size: int = 5
) -> None:
    """
    It displays the number of NAs, the percentage of NA, the number of unique values and
    the data type for each column.
    It can (depending on summary_stats param) also displays dataframe shape and
    also displays statistics for numerical variables.
    Finally, it displays the dataframe head or a random sample of dataframe according to user choice.

    Args
        dataframe: the pandas dataframe that the user wants to check.
        summary_stats: a boolean to indicate if user wants to see summary statistic
            for numerical features.
        head: boolean that indicate if user wants to see
            the head of the dataframe (True) or
            a sample of the dataframe (False)
        size: size of the dataframe.head() or dataframe.sample() function .

    Return
        None: a none type object
    """

    # import required libraris
    import pandas as pd
    import numpy as np

    #########################################################
    # This function requires that Jinja2 library is installed,
    # but it doesn't need to be imported
    # -> pip install Jinja2
    #########################################################

    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"summary_stats": summary_stats}, (bool,))
    validate_input_types({"head": head}, (bool,))
    validate_input_types({"size": size}, (int,))

    # ==========================
    # MEMORY USAGE INFORMATION &
    # DESCRIPTIVE INFORMATION &
    # SHAPE INFORMATION
    # ==========================
    check_na_unique_dtypes(dataframe)

    # ======= STATISTICS =======
    # check if user wants summary statistics
    if summary_stats:
        # use summary_statistics function of this same module
        print("-----------------------------")
        summary_statistics(dataframe)

    # ======= DATAFRAME INSTANCES =======

    print("-----------------------------")

    # check if user wants df.head()
    if head:
        print(f"\n\ndataframe.head({size})")
        display(dataframe.head(size))

    # user wants df.sample()
    else:
        print(f"\n\ndataframe.sample({size})")
        display(dataframe.sample(size))

    print("*" * 49)

    return None  # explicitly


def inspect_dtypes(dataframe: pd.DataFrame, n_samples: int = 5) -> None:
    """
    It displays the name of the columns (of the input dataframe),
    the type of the columns (of the input dataframe) and
    a random sample (of the input dataframe).

    Args
        dataframe: the pandas dataframe that the user wants to check.
        n_samples: an integer with the number of samples to display.

    Return
        None: a none type object
    """

    # import required libraries
    import pandas as pd

    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"n_samples": n_samples}, (int,))

    # get the data types for every column
    df_types = pd.DataFrame(dataframe.dtypes)
    # rename column with types
    df_types.columns = ["types"]

    # get a random sample of size "n_samples" and transpose the result
    df_samples = dataframe.sample(n_samples).T
    # rename columns
    df_samples.columns = [f"random row: {i+1}" for i in range(n_samples)]

    # merge dataframes on indexes
    df_dtypes = pd.merge(
        left=df_types, right=df_samples, left_index=True, right_index=True
    )

    # print descriptive data
    display(
        df_dtypes.style.map(
            lambda x: "background-color: Navy; color: White", subset=["types"]
        )
    )

    return None  # explicitly


def datetime_summary(dataframe: pd.DataFrame) -> None:
    """
    It numerically describes the datetime columns of the given dataframe.

    Args
        dataframe: a pandas dataframe whose datetime columns are to be described.

    Return
        None: a none type object
    """

    # import required libraris
    import pandas as pd
    import numpy as np

    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))

    # get datetime variables
    dataframe = dataframe.select_dtypes(include=["datetime", "datetime64[ns, UTC]"])

    # reuse pandas describe method and transpose the dataframe
    df_aux = dataframe.describe().T

    # create a range column: diference in months between the min and the max date
    df_aux["range [months]"] = (df_aux["max"] - df_aux["min"]).apply(lambda x: x.days) / 30.4375

    # create a column to indicate the number of unique values
    df_aux["nunique"] = dataframe.nunique(dropna=True)

    # drop columns refering to Q1 and Q3
    df_aux.drop(columns=["25%", "75%"], inplace=True)

    # calculate the number of NAs and the percentage of NAs for the given column
    df_aux["Num NAs"] = dataframe.isna().sum(axis=0)
    df_aux["Percent NAs"] = (dataframe.isna().mean(axis=0) * 100).apply(
        lambda x: int(np.ceil(x))
    )

    # rename columns
    df_aux.rename(
        columns={
            "count": "count [non-NA]",
            "50%": "median",
            "min": "first date",
            "max": "last date",
        },
        inplace=True,
    )

    # reorder columns
    df_aux = df_aux[
        [
            "first date",
            "last date",
            "range [months]",
            "mean",
            "median",
            "Num NAs",
            "Percent NAs",
            "count [non-NA]",
            "nunique",
        ]
    ]

    # open pandas options with context manager
    with pd.option_context("display.float_format", "{:,.2f}".format):
        # display dataframe highlight specific columns
        display(
            df_aux.style.map(
                lambda x: "background-color: Navy; color: White",
                subset=["first date", "last date", "range [months]", "Percent NAs"],
            )
        )

    return None  # explicitly


def complete_value_counts(
    dataframe: pd.DataFrame,
    column: str,
    display_results: bool = True,
) -> pd.DataFrame:
    """
    Calculate the absolute and the percentage value counts for the given column
    of the given dataframe

    Args
        dataframe: a pandas dataframe
        column: a string witht the column name to display value counts
        display_results: a boolean to indicate if user wants to display the dataframe (True)
            or only return it (False)

    Return
        df_value_counts: a pandas dataframe with absolute and percentage value counts
    """

    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"column": column}, (str,))

    # apply pandas value_counts function to the dataframe
    df_value_counts = pd.DataFrame(dataframe[column].value_counts(dropna=False))
    # reset index
    df_value_counts = df_value_counts.reset_index()
    # rename columns
    df_value_counts = df_value_counts.rename(
        columns={"index": "label", column: f"{column}_absolute"}
    )
    # create a percentage column besides absolute value_counts
    df_value_counts[f"{column}_percent"] = (
        df_value_counts[f"{column}_absolute"]
        / df_value_counts[f"{column}_absolute"].sum()
    )
    # get percentage column in %
    df_value_counts[f"{column}_percent"] = df_value_counts[f"{column}_percent"] * 100
    # sort final dataframe by descending percent column
    df_value_counts.sort_values(
        by=f"{column}_percent",
        axis=0,
        ascending=False,
        inplace=True,
        na_position="last",
        ignore_index=True,
    )

    # check if user wants results to be displayed
    if display_results:
        # display value counts
        print(
            "-" * 49,
            f"\nAbsolute and percentual value counts for {column.upper()} column.\n",
        )
        display(df_value_counts)
        print("-" * 49)

    return df_value_counts


def check_duplicating_columns(
    dataframe: pd.DataFrame,
    df_granularity: list,
    display_results: bool = True,
) -> list:
    """
    Display the columns with more than one unique value for the dataframe granularity

    Args
        dataframe: a pandas dataframe
        df_granularity: a list with the name of the columns that compose the dataframe granularity
        display_results: a boolean to indicate if duplicated columns
            shold be displayed in a dataframe format

    Return
        duplicating_columns: a list with the names of the columns that are duplicating granularity
    """

    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"df_granularity": df_granularity}, (list,))
    validate_input_types({"display_results": display_results}, (bool,))

    # get the number of unique rows for each dataframe granularity
    df_nunique_grain = dataframe.groupby(df_granularity).nunique()

    # get the maximum number of unique rows per column
    df_nunique_grain_max = df_nunique_grain.max(axis=0)

    # define column name
    col_name = "Number of columns unique values per dataframe granularity"

    # create a dataframe to make it easier to understand
    df_duplicating_columns = pd.DataFrame(
        data=df_nunique_grain.max(axis=0),
        columns=[col_name],
    )

    # create a pandas series with highlighting condition
    highlight_series_cond = df_duplicating_columns[col_name] > 1

    # get highlighting map
    highlighted_rows = highlight_series_cond.map(
        {True: "background-color: Navy; color: White", False: ""}
    )

    # check if user wants duplicating columns to be displayed in dataframe format
    if display_results:
        # display highlighted dataframe
        display(df_duplicating_columns.style.apply(lambda _: highlighted_rows))

    # get the names of the columns that are duplicating dataframe granularity
    duplicating_columns = df_duplicating_columns[
        df_duplicating_columns[col_name] > 1
    ].index.tolist()

    return duplicating_columns


def check_float_series_is_int(dataframe: pd.DataFrame, column: str) -> list:
    """
    Check how many row in the given column of the given dataframe that has
    some decimals. It returns the index of the values whose modulo over 1
    is different than zero as well as printing relevant info.

    Args
        dataframe: a pandas dataframe.
        column: a string with the name of the column
            to check decimal on the given dataframe.

    Return
        modulo_index: a list with the index of the values whose modulo over 1
            is different than zero as well as p
    """

    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_dataframe_cols(dataframe, (column,))

    # create a pandas series to inspect values
    series = dataframe[column]

    # calculate the modulo of the series over 1
    series_modulo = series.mod(1, fill_value=None)
    # get a boolean series to indicate if modulo is greater than zero
    series_bool = series_modulo > 0
    # get the total number of values whose modulo is different than zero
    sum_series_bool = series_bool.sum()
    # check if some value has modulo differente than one
    if sum_series_bool > 0:
        # print report
        print(
            f"\n\n{'*'*49}\n"
            f"There are {sum_series_bool} values on the column {column} of the given dataframe that have some decimals"
            f"\n\n{'*'*49}\n"
        )

        # get the index whose modulo are different than zero
        modulo_index = series_bool[series_bool].index.tolist()

        return modulo_index

    # all values have modulo equal to zero
    else:
        # print report
        print(
            f"\n{'*'*49}\n\n"
            f'All values in the column "{column}" of the given dataframe have NO decimals!'
            f"\n\n{'*'*49}\n"
        )


def check_dtype_convertion(inspection_list: list, to_dtype_list: list) -> dict:
    """
    Take a list of input values and check what values in the list are not convertible
    to the input types on param to_dtype_list.

    Args
        inspection_list: a list with values to inspect convertion
        to_dtype_list: a list with dtypes to check convertion.
            Must be: int, float, bool or str

    Return
        dtypes_report: a dictionary with reporting information regarding convertion check.
    """

    # input verification
    validate_input_types({"inspection_list": inspection_list}, (list,))
    validate_input_types({"to_dtype_list": to_dtype_list}, (list,))
    for item in to_dtype_list:
        validate_input_values(
            {"dtype": item},
            (int, float, bool, str),
            "to_dtypes items must be int, float, bool or str",
        )

    # create a dictionary to hold final report
    dtypes_report = {}

    # iterate over dtypes_list
    for to_dtype in to_dtype_list:
        # create a empty set to hold exception values
        exception_set = set()

        # iterate over unique values in list
        for i in set(inspection_list):
            # try to convert value
            try:
                to_dtype(i)
            # in case of errors
            except ValueError:
                exception_set.add(i)

        # check if no exception was raised
        if len(exception_set) == 0:
            # print report
            print(
                f"{'*'*49}\n"
                f"All values of the given input are convertible to type {to_dtype}\n"
                f"{'*'*49}\n"
            )
            # add value to dtypes_report dict
            dtypes_report[str(to_dtype)] = []

        # some exception was raised
        else:
            # print report
            print(
                f"{'*'*49}\n"
                f"The following values are NOT convertible to type {to_dtype}:\n"
                f"{list(exception_set)}\n"
                f"{'*'*49}\n"
            )
            # add value to dtypes_report dict
            dtypes_report[str(to_dtype)] = [exception_set]

    return dtypes_report


def custom_print(
    text: str, 
    font_style: str = "normal", 
    text_color: str = "black",
    background_color: str = None
) -> None:
    """
    It takes the text input and print it with the required font styles, 
    font colors and background colors.

    Args
        text: a string with the text to print.
        font_style: a string with the required font style
            Available styles are "normal", "bold", "light", 
            "italic", "underline" and "blink".
        text_color: a string with the required text color.
            Available colors are "black", "red", "green", "yellow", 
            "blue", "purple", "cyan" and "white".
        background_color: a string with the required background color
            Available colors are "black", "red", "green", "yellow", 
            "blue", "purple", "cyan" and "white",

    Return
        None: a NoneType object
    """

    # input verification
    validate_input_types({"text": text}, (str,))
    validate_input_types({"style": font_style}, (str,))
    validate_input_values(
        {"style": font_style},
        ("normal", "bold", "light", "italic", "underline", "blink")
    )
    validate_input_types({"text_color": text_color}, (str,))
    validate_input_values(
        {"text_color": text_color},
        ("black", "red", "green", "yellow", "blue", "purple", "cyan", "white")
    )

    if background_color is not None:
        validate_input_types({"background_color": background_color}, (str,))
        validate_input_values(
            {"background_color": background_color},
            ("black", "red", "green", "yellow", "blue", "purple", "cyan", "white")
        )

    # define escape codes to be used
    ESCSEQ_unicode = "\u001b["
    ESCSEQ_octal = "\033["
    
    # define font styles available
    STYLE_MAP = {
        "normal": 0,
        "bold": 1,
        "light": 2,
        "italic": 3,
        "underline": 4,
        "blink": 5
    }

    # define colors available
    COLOR_MAP_TEXT = {
        "black": ";30", 
        "red": ";31",
        "green": ";32",
        "yellow": ";33",
        "blue": ";34",
        "purple": ";35",
        "cyan": ";36",
        "white": ";37"
    }
    COLOR_MAP_BACK = { # add 10 to numbers in COLOR_MAP_TEXT values
        name: f";{int(code[1:]) + 10}" for name, code in COLOR_MAP_TEXT.items()
    }

    # define block variables to print correctly
    BACK_COLOR_BLOCK = COLOR_MAP_BACK.get(background_color, "")
    TEXT_COLOR_BLOCK = COLOR_MAP_TEXT.get(text_color, "")
    STYLE_BLOCK = STYLE_MAP[font_style]

    # construct string and print
    print(
        f"{ESCSEQ_octal}{STYLE_BLOCK}{TEXT_COLOR_BLOCK}{BACK_COLOR_BLOCK}m{text}\033[0;0m"
    )