##############################
####### INITIAL CONFIG #######
##############################

# import required project modules
from project_lib.input_validation import validate_input_types


################################
####### MODULE FUNCTIONS #######
################################


def initial_settings() -> None:
    """
    Set initial settings for dataframes and plotting diplays
    """
    ####################
    # import libraries #
    ####################
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from IPython.display import display, HTML

    #####################
    # pandas dataframes #
    #####################
    # Define pandas config to avoid overwriting dfs
    pd.options.mode.copy_on_write: bool = True
    # set cientific notation for pandas
    pd.set_option(
        "display.float_format", "{:,.3f}".format
    )  # used in some places like SeriesFormatter
    pd.set_option(
        "display.precision", 3
    )  # for regular formatting as well as scientific notation
    pd.set_option(
        "styler.format.precision", 3
    )  # The precision for floats and complex numbers
    # don't truncate columns
    pd.set_option("display.max_colwidth", 100)  # None for unlimited
    # display all columns
    pd.set_option("display.max_columns", None)
    # display up to 100 rows
    pd.set_option("display.max_rows", 100)
    # display dimensions
    pd.set_option("display.show_dimensions", True)
    # define decimals and thousand separation
    pd.set_option("styler.format.decimal", ",")
    pd.set_option("styler.format.thousands", ".")

    ####################
    # matplotlib plots #
    ####################

    # set default plt figure size
    plt.rcParams["figure.figsize"] = [10, 5]
    # figure suptitle
    plt.rcParams["figure.titlesize"] = "large"
    plt.rcParams["figure.titleweight"] = "bold"
    # set default plt font size
    plt.rcParams["font.size"] = 24
    # font weight
    # plt.rcParams["font.weight"] = "bold"
    # title location
    # plt.rcParams["axes.titlelocation"] = "left"
    # title size
    plt.rcParams["axes.titlesize"] = "large"
    # title wight
    plt.rcParams["axes.titleweight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    # spines
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    # axis labels
    # plt.rcParams["xaxis.labellocation"] = "left"
    # plt.rcParams["yaxis.labellocation"] = "top"
    # figure layout
    plt.rcParams["figure.autolayout"] = False
    # save figures
    plt.rcParams["savefig.dpi"] = 900
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.format"] = "png"

    # set figures to seaborn style
    plt.style.use("fivethirtyeight")

    #####################
    # jupyter notebooks #
    #####################

    # set cell size to be expanded
    display(HTML("<style>.container { width:100% !important; }</style>"))