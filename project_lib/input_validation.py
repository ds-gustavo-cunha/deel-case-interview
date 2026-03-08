##############################
####### INITIAL CONFIG #######
##############################

# import required library to configure module

# import required libraries
import pandas as pd


################################
####### MODULE FUNCTIONS #######
################################


def validate_input_types(
    dict_values: dict, enforce_type: tuple, error_msg: str = None
) -> None:
    """
    This function gets the dictionary with param_name-param_value pairs
    and check if param_value values are of type enforce_type

    Args
        dict_values: a dictionary with param_name-param_value pairs.
        enforce_type: a tuple with types of param_value values
        error_msg: a string with the error message in case of errors
            if user wants a custom message

    Return:
        None
    """

    #####################################
    ####### VALIDATE INPUT PARAMS #######
    #####################################

    # check if dict_values is a dictionary
    if not isinstance(dict_values, dict):
        # raise error
        raise TypeError(f"dict_values param must be of type: dict")
    # check if the length of dict_values is one
    if len(dict_values) != 1:
        # raise error
        raise ValueError(
            f"dict_values param must be a dict with only one key-value pair"
        )
    # check if enforce_type is a tuple
    if not isinstance(enforce_type, tuple):
        # raise error
        raise TypeError(f"enforce_type param must be of type: tuple")
    # iterate over items in enforce_type
    for item in enforce_type:
        # check if items of enforce_type are of type "type"
        if not isinstance(item, type):
            # raise error
            raise TypeError(f"items of enforce_type param must be of type: type")
    # check if error message was input
    if (error_msg is not None) and (not isinstance(error_msg, str)):
        # raise error
        raise ValueError(f"error_msg param must be of type: str")

    ######################################
    ####### VALIDATE PARAMS VALUES #######
    ######################################

    # iterate over values in dict_values
    for key, value in dict_values.items():
        # check if value is of type enforce_type
        if not isinstance(value, enforce_type):
            # check if error message was input
            if error_msg is not None:
                # raise TypeError with custom message
                raise TypeError(error_msg)
            # error message was not input
            else:
                # raise TypeError with default message
                raise TypeError(f"param {key} must be of types: {list(enforce_type)}")

    return None  # explicitly


def validate_input_values(
    dict_values: dict, enforce_values: tuple, error_msg: str = None
) -> None:
    """
    This function gets the dictionary with param_name-param_value pairs
    and check if param_value values are of type enforce_values

    Args
        dict_values: a dictionary with param_name-param_variable pairs
            Ex.: "param_name_as_string": param_variable
        enforce_values: a tuple with types of param_value values
            Ex.: (1, 2, 3)
        error_msg: a string with the error message in case of errors
            if user wants a custom message

    Return:
        None
    """

    #####################################
    ####### VALIDATE INPUT PARAMS #######
    #####################################

    # check if dict_values is a dictionary
    if not isinstance(dict_values, dict):
        # raise error
        raise TypeError(f"dict_values param must be of type: dict")
    # check if the length of dict_values is one
    if len(dict_values) != 1:
        # raise error
        raise TypeError(
            f"dict_values param must be a dict with only one key-value pair"
        )
    # check if enforce_type is a tuple
    if not isinstance(enforce_values, tuple):
        # raise error
        raise TypeError(f"enforce_values param must be of type: tuple")
    # check if error message was input
    if (error_msg is not None) and (not isinstance(error_msg, str)):
        # raise error
        raise ValueError(f"error_msg param must be of type: str")

    ######################################
    ####### VALIDATE PARAMS VALUES #######
    ######################################

    # iterate over values in dict_values
    for key, value in dict_values.items():
        # check if value is in enforce_values
        if value not in enforce_values:
            # check if error message was input
            if error_msg is not None:
                # raise ValueError with custom message
                raise ValueError(error_msg)
            # error message was not input
            else:
                # raise ValueError
                raise ValueError(f"param {value} is not in {list(enforce_values)}")

    return None  # explicitly


def validate_dataframe_cols(
    dataframe: pd.DataFrame, columns: tuple, error_msg: str = None
) -> None:
    """
    This function gets a tuple with columns and check if
    all of them are valid column on the given dataframe

    Args
        dict_values: a pandas dataframe.
        columns: a tuple with the column names
        error_msg: a string with the error message in case of errors
            if user wants a custom message

    Return:
        None
    """

    #####################################
    ####### VALIDATE INPUT PARAMS #######
    #####################################

    # check if dataframe params  is a dictionary
    if not isinstance(dataframe, pd.core.frame.DataFrame):
        # raise error
        raise TypeError(f"dataframe param must be of type: pandas.core.frame.DataFrame")
    # check if cols is a tuple
    if not isinstance(columns, tuple):
        # raise error
        raise TypeError(f"columns param param must be of type: tuple")

    ######################################
    ####### VALIDATE PARAMS VALUES #######
    ######################################

    # iterate over cols param
    for col in columns:
        # check if col is a column on the dataframe
        if col not in dataframe.columns:
            # check if error message was input
            if error_msg is not None:
                # raise ValueError with custom message
                raise ValueError(error_msg)
            # error message was not input
            else:
                # raise ValueError
                raise ValueError(f"Dataframe has no column named: {col}")

    return None  # explicitly


# def validate_inputs(dict_values: dict, enforce_type: tuple) -> None:
#     """
#     This function gets the dictionary with param_name-param_value pairs
#     and check if param_value values are of type enforce_type

#     Args
#         dict_values: a dictionary with param_name-param_value pairs.
#         enforce_type: a tuple with types of param_value values

#     Return:
#         None
#     """

#     ##########################################
#     ####### VALIDATE "INTERNAL" PARAMS #######
#     ##########################################
#     # check if dict_values is a dictionary
#     if not isinstance(dict_values, dict):
#         # raise error
#         raise TypeError(f"dict_values param must be of type: dict")
#     # check if enforce_type is a tuple
#     if not isinstance(enforce_type, tuple):
#         # raise error
#         raise TypeError(f"enforce_type param must be of type: tuple")
#     # iterate over items in enforce_type
#     for item in enforce_type:
#         # check if items of enforce_type are of type "type"
#         if not isinstance(item, type):
#             # raise error
#             raise TypeError(f"items of enforce_type param must be of type: type")

#     ##########################################
#     ####### VALIDATE "EXTERNAL" PARAMS #######
#     ##########################################

#     # iterate over values in dict_values
#     for key, value in dict_values.items():
#         # check if value is of type enforce_type
#         if not isinstance(value, enforce_type):
#             # raise TypeError
#             raise TypeError(f"{key} param must be of types: {list(enforce_type)}")

#     return None  # explicitly


# # NEW VALIDATION MODULE FUNCTIONS
