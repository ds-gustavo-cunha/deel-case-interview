##############################
####### INITIAL CONFIG #######
##############################

# import required library to configure module
import pandas as pd
from project_lib.initial_config import initial_settings
from project_lib.input_validation import validate_input_types, validate_dataframe_cols

# set the basic cofiguration for this module
initial_settings()

################################
####### MODULE FUNCTIONS #######
################################


def check_for_bias(
    dataframe: pd.DataFrame,
    treatment: str,
    showfliers: bool = False,
    figsize: tuple = None,
    num_cols: int = 3,
    saving_path: str = None,
    top_n_categs: int = 10,
    max_num_cat: int = 15,
) -> None:
    """
    Iterate over combinations of outcome variable and dataframe columns
    and plot over these combinations to check for bias before AB testing

    Args
        dataframe: a pandas DataFrame with the data to check for bias
        treatment: a str with the treatment variable column name
        showfliers: a boolean to indicate whether to show (or not) outliers on boxplot
        figsize: a tuple with the figsize to plot
        num_cols: an int with the number of columns to plot variables
        saving_path: a string with the path to save figure
        top_n_categs: an int with the maximum number of top categories (by frequency) to show
        max_num_cat: an int with the maximum number of characters for category labels
    """
    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"outcome_variable": treatment}, (str,))
    validate_input_types({"showfliers": showfliers}, (bool,))
    validate_dataframe_cols(dataframe, (treatment,))
    if figsize is not None:
        validate_input_types({"figsize": figsize}, (tuple,))
    validate_input_types({"num_cols": num_cols}, (int,))
    if saving_path is not None:
        validate_input_types({"saving_path": saving_path}, (str,))
    validate_input_types({"top_n_categs": top_n_categs}, (int,))
    validate_input_types({"max_num_cat": max_num_cat}, (int,))

    # import required libraries
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import seaborn as sns
    from scipy.stats import linregress, chi2_contingency, kruskal

    # define plot style
    plt.style.use("fivethirtyeight")

    # define columns according to dtypes
    numeric_cols = dataframe.select_dtypes(include=['number']).columns.tolist()
    categ_cols = dataframe.select_dtypes(include=['object']).columns.tolist()
    date_cols = dataframe.select_dtypes(include=['datetime']).columns.tolist()

    # get columns that are not outcome variable
    non_treatment = numeric_cols + categ_cols
    non_treatment.remove(treatment)
    non_treatment

    # define number of rows
    n_rows = dataframe.shape[1] // num_cols + 1

    # check if user input figsize
    if figsize is None:
        # assign th default figsize
        figsize = (num_cols * 7, n_rows * 6)

    # create a figure object
    fig = plt.figure(figsize=figsize, tight_layout=True)

    # create grid for plotting
    specs = gridspec.GridSpec(ncols=num_cols, nrows=n_rows, figure=fig)

    # get number of plots
    num_plots = len(non_treatment)
        
    # define orientation of boxplot
    orient = {True:"v", False:"h"}
        
    #  iterate over combination of outcome and each other variable
    for index, outcome, non_outcome in (
        zip(range(0, num_plots), [treatment]*num_plots, non_treatment)
    ):
        # create a subplot to plot the given feature
        ax = fig.add_subplot(specs[index // num_cols, index % num_cols])

        # limit categorical columns to top_n_categs categories by frequency and truncate labels
        top_categ_suffix = ""
        if non_outcome in categ_cols:
            top_categs = dataframe[non_outcome].value_counts().nlargest(top_n_categs).index
            df_plot_data = dataframe[dataframe[non_outcome].isin(top_categs)].copy()
            df_plot_data[non_outcome] = df_plot_data[non_outcome].str[:max_num_cat]
            if dataframe[non_outcome].nunique() > top_n_categs:
                top_categ_suffix = f" [top {top_n_categs}]"
        else:
            df_plot_data = dataframe

        # check if both plotting variables are numeric ---> regression plot
        if (outcome in numeric_cols) and (non_outcome in numeric_cols):
            # plot regression over scatter plot
            rp = sns.regplot(
                data=df_plot_data, x=non_outcome, y=outcome,
                scatter=True, fit_reg=True, ci=95, n_boot=1000,
                line_kws={"color": "red", "linewidth":1, "linestyle":"--"},
                ax=ax
            )
            # calculate slope and intercept of regression equation
            try:
                slope, intercept, rvalue, pvalue, sterr = linregress(
                    x=df_plot_data[non_outcome],
                    y=df_plot_data[outcome],
                    alternative="two-sided"
                )
                reg_label = (
                    f"Reg slope = {slope:.3f} [p-value = {pvalue:.3f}]"
                )
            except ValueError:
                reg_label = "Reg slope [p-value]: N/A (constant col)"

            # define title
            ax.set_title(f"Treatment: {outcome}\nVariable: {non_outcome}\n{reg_label}")

        # check if both plotting variables are categorical ---> bar plot
        elif (outcome in categ_cols) and (non_outcome in categ_cols):
            # groupby dataframe by variables to plot and get size
            df_plot = df_plot_data.groupby(by=[non_outcome, outcome], as_index=False).size() # not count NaNs
            # plot a bar chart
            sns.barplot(
                data=df_plot, x=non_outcome, y="size",
                hue=outcome, edgecolor=".0",
                ax=ax
            )
            # run chi-squared test of independence on contingency table
            try:
                contingency_table = pd.crosstab(df_plot_data[non_outcome], df_plot_data[outcome])
                _, chi2_pvalue, _, _ = chi2_contingency(contingency_table)
                chi2_label = f"\nChi² p-value: {chi2_pvalue:.3f}"
            except ValueError:
                chi2_label = "\nChi² p-value: N/A"
            # define title
            ax.set_title(f"Treatment: {outcome}\nVariable: {non_outcome}{top_categ_suffix}{chi2_label}")

        # one variable is numeric while the other is categoric ---> box plot
        else:
            # plot a box plot
            sns.boxplot(
                data=df_plot_data, x=non_outcome, y=outcome,
                orient=orient[outcome in numeric_cols],
                meanline=True, showmeans=True, meanprops={"color": "black", "marker": "*"},
                showfliers=showfliers,
                ax=ax
            )
            # identify numeric and categorical variables for Kruskal-Wallis test
            num_var, cat_var = (outcome, non_outcome) if outcome in numeric_cols else (non_outcome, outcome)
            # run Kruskal-Wallis test: compare numeric variable across groups of categorical variable
            try:
                groups = [group[num_var].dropna().values for _, group in df_plot_data.groupby(cat_var)]
                _, kruskal_pvalue = kruskal(*groups)
                kruskal_label = f"\nKruskal-Wallis p-value: {kruskal_pvalue:.3f}"
            except ValueError:
                kruskal_label = "\nKruskal-Wallis p-value: N/A"
            # define title
            ax.set_title(f"Treatment: {outcome}\nVariable: {non_outcome}{top_categ_suffix}\nShow outliers: {showfliers}{kruskal_label}")

        # check if x axis has many categories
        if non_outcome in categ_cols:
            # rotate x labels
            ax.tick_params(axis="x", rotation=90)

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

    # display plot
    plt.show(); 