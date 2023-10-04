# ### This module contain all functions related to the cheking user inputs ###

# ## Imports
# # from standard python modules

# from third parties
import numpy as np
# import pandas as pd
from matplotlib.axes import SubplotBase
# self made



# ## List of functions with TESTS

# ## List of functions WITHOUT TESTS

# # - _check_a_lower_than_b(value_a, value_b, param_a_name, param_b_name)
# # - _check_column_name_in_dataframe(name, param_name1, dataframe, param_name2)
# # - _check_data_in_range(value, param_name, lower, upper)
# # - _check_forbidden_character(value, param_name)
# # - _check_is_bool(value, param_name)
# # - _check_is_dict(value, param_name)
# # - _check_is_data_frame(df, param_name,)
# # - _check_value_is_equal_or_higher_than(value, param_name, minimum)
# # - _check_is_float(value, param_name)
# # - _check_is_float_or_int(value, param_name)
# # - _check_is_integer(value, param_name)
# # - _check_is_list(value, param_name)
# # - _check_is_numpy_1_D(value, param_name)
# # - _check_is_positive(value, param_name)
# # - _check_is_str(value, param_name)
# # - _check_is_subplots(value, param_name)
# # - _check_list_of_types(lista, param_name, expected_type)
# # - _check_list_size(lista, param_name, minimum)
# # - _check_matching_sample_size(arr1, param_name1, arr2, param_name2)
# # - _check_param_values(value, param_name, param_values)
# # - _check_sample_size(arr, param_name, minimum)
# # - _convert_to_one_D_numpy(arr, param_name)



# ## Functions


def _check_is_float_or_int(value, param_name):
    """This function checks if a ``value`` is an ``float`` or ``int``.

    Parameters
    ----------
    value : any type
        The value to check if it is a ``float`` or ``int``.
    param_name : ``str``
        The original name of the parameter passed through the parameter ``value``.


    Returns
    -------
    ``True`` if ``value`` is ``float`` or ``int``.
    Raises ``ValueError`` if ``value`` is not a ``float`` or ``int``.

    """

    if isinstance(value, (int, np.uint, np.integer, float, np.floating)) == False:
        try:
            raise ValueError("Not a number")
        except ValueError:
            print(f"\n\nThe parameter {param_name} must be numeric, but it is of type '{type(value).__name__}'.\n\n")
            raise
    else:
        return True


# def _check_column_name_in_dataframe(name, param_name1, dataframe, param_name2):
#     """This function checks if the ``name`` is a valid column name for the ``dataframe``.

#     Parameters
#     ----------
#     name : ``str``
#         The column name to be checked.
#     param_name1 : ``str``
#         The original name of the parameter passed through the parameter ``name``.
#     dataframe : ``pd.DataFrame``
#         The data frame to check the name.
#     param_name2 : ``str``
#         The original name of the parameter passed through the parameter ``dataframe``.

#     Returns
#     -------
#     ``True`` if ``name`` is a DataFrame column_name.
#     Raises ``ValueError`` otherwise.


#     """

#     if name not in dataframe.columns:
#         try:
#             raise TypeError("Column name not found on DataFrame")
#         except TypeError:
#             print(f"\n\n The DataFrame does not have a column with the name '{name}'.\n\n")
#             raise
#     return True



def _check_data_in_range(value, param_name, lower, upper):
    """This function checks if a ``value`` is within the range between lower and upper.

    Parameters
    ----------
    value : ``int`` or ``float``
        The value to be evaluated
    param_name : ``str``
        The original name of the parameter passed through the parameter ``value``.
    lower : ``int`` or ``float``
        The lower bound
    upper : ``int`` or ``float``
        The upper bound


    Notes
    -----
    If ``lower`` is higher than ``upper``, the function corrects these values automatically.

    Returns
    -------
    ``True`` if ``value`` is in the range: ``min < value < max``
    ``ValueError`` if ``value`` is not in the range: ``min < value < max``

    """

    values = [lower, upper]
    lower = min(values)
    upper = max(values)



    if (lower < value < upper) == False:
        try:
            raise ValueError("Value out of limits")
        except ValueError:
            print(f"\n\nThe parameter '{param_name}' must be between '{lower}' and '{upper}', but the received value is equal to '{value}'\n\n")
            raise
    return True





# def _check_forbidden_character(value, param_name):
#     """This function checks if there are characters that can be problematic for a file name.

#     Parameters
#     ----------
#     value : string
#         The value to check if it has some espcific characters.
#     param_name : string
#         The original name of the parameter passed through the parameter 'value'.

#     Notes
#     -----

#     This function checks if a string contains some characters that can be problematic for saving files.
#     The forbidden characters are:

#         "/": 'forward slash',
#         "<": "less than",
#         ">": "greater than",
#         ":": "colon",
#         "\"": "double quote",
#         "\\": "back slash",
#         "|": "vertical bar",
#         "?": "question mark",
#         "*": "asterisk",
#         ".": "dot",
#         ",": "comma",
#         "[": "left square bracket",
#         "]": "right square bracket",
#         ";": "semicolon",

#     Returns
#     -------
#     True if value does not have any forbidden.
#     Raises ValueError is value is not a valid string.

#     References
#     ----------
#     .. [1] https://stackoverflow.com/a/31976060/17872198
#     .. [2] https://stackoverflow.com/q/1976007/17872198
#     """
#     list_of_forbbiden = {
#         "/": 'forward slash',
#         "<": "less than",
#         ">": "greater than",
#         ":": "colon",
#         "\"": "double quote",
#         "\\": "back slash",
#         "|": "vertical bar",
#         "?": "question mark",
#         "*": "asterisk",
#         ".": "dot",
#         ",": "comma",
#         "[": "left square bracket",
#         "]": "right square bracket",
#         ";": "semicolon",
#     }
#     for key in list_of_forbbiden.keys():
#         if key in value:
#             try:
#                 raise ValueError("Character not allowed")
#             except ValueError:
#                 print(f"\n\n The character '{key}' is not a valid character for the parameter {param_name}.\n\n")
#                 raise
#     return True



def _check_is_bool(value, param_name):
    """This function checks if a ``value`` is a boolean.

    This function verifies if the parameter ``value`` is the type of ``bool`` (``True`` or ``False``). If so, it returns ``True``. If it is not, the function raises a ``ValueError``.

    Parameters
    ----------
    value : any type
        The value to be evaluated.
    param_name : ``str``
        The original name of the parameter passed through the parameter ``value``.

    Returns
    -------
    ``True`` if ``value`` is a ``bool``
    ``ValueError`` if ``valuew`` is not a ``bool``


    """
    if isinstance(value, bool) == False:
        try:
            raise TypeError("Not a boolean")
        except TypeError:
            print(f"\n\n The parameter '{param_name}' must be of type 'bool', but its type is '{type(value).__name__}'.\n\n")
            raise

    return True



# def _check_is_dict(value, param_name):
#     """This function checks if a ``value`` is a ``dict``.


#     Parameters
#     ----------
#     value : any type
#         The value to check if it is a ``dict``.
#     param_name : ``str``
#         The original name of the parameter passed through the parameter ``value``.


#     Returns
#     -------
#     ``True`` if ``value`` is a ``dict``
#     Raises ``TypeError`` if ``value`` is not a ``dict``

#     """
#     if isinstance(value, dict) == False:
#         try:
#             raise TypeError("Not a dictionary")
#         except TypeError:
#             print(f"\n\n The parameter '{param_name}' must be of type 'dict', but its type is '{type(value).__name__}'.\n\n")
#             raise
#     return True

def _check_value_is_equal_or_higher_than(value, param_name, minimum):
    """This function checks if a ``value`` is equal or higher than ``minimum``.

    Parameters
    ----------
    value : ``int`` or ``float``
        The value to be evaluated
    param_name : ``str``
        The original name of the parameter passed through the parameter ``value``.
    minimum : ``int`` or ``float``
        the lower bound (closed)

    Returns
    -------
    ``True`` if ``value`` is equal or higher than ``minimum``.
    ``ValueError`` if ``value`` is lower than ``minimum``.

    """
    _check_is_float_or_int(value, "value")
    _check_is_float_or_int(minimum, "minimum")

    if value < minimum:
        try:
            raise ValueError("Out of bounds")
        except ValueError:
            print(f"\n\n The value of the parameter '{param_name}' must be greater than '{minimum}', but it is ('{value}').\n\n")
            raise
    return True

# def _check_is_data_frame(df, param_name,):
#     """This function checks if ``df`` is a valid ``DataFrame``, e.g., if it is ``DataFrame`` and if it is not empty.

#     Parameters
#     ----------
#     df : any type
#         The value to check if it is a ``DataFrame``.
#     param_name : ``str``
#         The original name of the parameter passed through the parameter ``df``.


#     Returns
#     -------
#     ``True`` if ``df`` is a valid ``DataFrame``.
#     Raises ``ValueError`` is ``df`` is not a valid ``DataFrame``.


#     """

#     if isinstance(df, pd.DataFrame) == False:
#         try:
#             raise TypeError("Not a DataFrame")
#         except TypeError:
#             print(f"\n\n The parameter '{param_name}' must be of type 'DataFrame', but its type is '{type(value).__name__}'.\n\n")
#             raise
#     if df.empty:
#         try:
#             raise TypeError("Empty DataFrame")
#         except TypeError:
#             print(f"\n\nThe given dataframe is empty!\n\n")
#             raise
#     return True



# def _check_is_float(value, param_name):
#     """This function checks if a ``value`` is an ``float``

#     Parameters
#     ----------
#     value : any type
#         The value to check if it is a ``float``.
#     param_name : ``str``
#         The original name of the parameter passed through the parameter ``value``.


#     Returns
#     -------
#     ``True`` if ``value`` is ``float``
#     Raises ``TypeError`` if ``value`` is not a ``float``


#     """

#     if isinstance(value, (float, np.floating)) == False:
#         try:
#             raise TypeError("Not a float value")
#         except TypeError:
#             print(f"\n\n The parameter '{param_name}' must be of type 'float', but its type is '{type(value).__name__}'.\n\n")
#             raise
#     else:
#         return True



def _check_is_integer(value, param_name):
    """This function checks if a ``value`` is an ``int``

    Parameters
    ----------
    value : any type
        The value to check if it is an ``int``.
    param_name : ``str``
        The original name of the parameter passed through the parameter ``value``.

    Returns
    -------
    ``True`` if ``value`` is ``int``
    Raises ``TypeError`` if ``value`` is not an ``int``

    """
    if isinstance(value, (int, np.uint, np.integer)) == False:
        try:
            raise TypeError("Not an integer Error")
        except TypeError:
            print(f"\n\n The parameter '{param_name}' must be type of 'int', but it is '{type(value).__name__}'. \n\n")
            raise
    return True

def _check_is_list(value, param_name):
    """This function checks if a ``value`` is a ``list``.

    Parameters
    ----------
    value : any type
        The value to check if it is a ``list``.
    param_name : ``str``
        The original name of the parameter passed through the parameter ``value``.

    Returns
    -------
    ``True`` if ``value`` is a ``list``
    Raises ``TypeError`` if ``value`` is not a ``list``

    """
    if isinstance(value, list) == False:
        try:
            raise TypeError("Not a list error")
        except TypeError:
            print(f"\n\n The parameter '{param_name}' must be type of 'list', but it is '{type(value).__name__}'. \n\n")
            raise
    return True



def _check_a_lower_than_b(value_a, value_b, param_a_name, param_b_name):
    """This function checks if a ``value_a`` is lower than ``value_b``.

    Parameters
    ----------
    value_a : number
        The lower value
    value_b : number
        The upper value        
    param_a_name : ``str``
        The original name of the parameter passed through the parameter ``value_a``.
    param_b_name : ``str``
        The original name of the parameter passed through the parameter ``value_b``.

    Returns
    -------
    ``True`` if value_a lower than value_b
    Raises ``ValueError`` if if value_a greater or equto to value_b

    """
    if value_a >= value_b:
        try:
            raise ValueError("Range mismatch error")
        except ValueError:
            print(f"\n\n The parameter '{param_a_name}' must have a value smaller than the value of the parameter {param_b_name}, but {value_a}>={value_b}.\n\n")
            raise
    else:
        return True    



def _check_is_numpy_1_D(value, param_name):
    """This function checks if a ``value`` is an ``numpy array`` of 1 dimension

    Parameters
    ----------
    value : any
        The value to check if it is a non-empty 1-dimensional numpy array.
    param_name : ``str``
        The original name of the parameter passed through the parameter ``value``.


    Returns
    -------
    ``True`` if ``value`` is a non-empty ``1-dimensional numpy array``
    Raises ``ValueError`` if ``value`` is not a non-empty ``1-dimensional numpy array``

    """

    if isinstance(value, np.ndarray) == False:
        try:
            raise ValueError("Not a NumPy array Error")
        except ValueError:
            print(f"\n\n The parameter '{param_name}' must be a numpy array, but it is of type '{type(value).__name__}'\n\n")
            raise
    elif value.ndim != 1:
        try:
            raise ValueError("Dimension mismatch")
        except ValueError:
            print(f"\n\n The parameter '{param_name}' must contain one dimension, but ndim = '{value.ndim}'")
            raise
    elif value.size == 0:
        try:
            raise ValueError("Empty array error") 
        except ValueError:
            print(f"Parameter '{param_name}' cannot be an empty array")
            raise
    else:
        return True


def _check_is_positive(value, param_name):
    """This function checks if ``value`` is a positive number.

    Parameters
    ----------
    value : any
        The value to be tesed if it is a positive number
    param_name : ``str``
        The original name of the parameter passed through the parameter ``value``.

    Returns
    -------
    ``True`` if ``value`` is positive
    Raises ``ValueError`` if ``value`` is not positive


    """
    if value <= 0:
        try:
            raise ValueError("Not a positive value Error")
        except ValueError:
            print(f"\n\nThe parameter '{param_name}' must be a positive number, but it equals '{value}'\n\n")
            raise
    return True



def _check_is_str(value, param_name):
    """This function checks if a ``value`` is a ``str``.

    Parameters
    ----------
    value : any type
        The value to check if it is a ``str``.
    param_name : ``str``
        The original name of the parameter passed through the parameter ``value``.

    Returns
    -------
    ``True`` if value is a valid ``str``.
    Raises ``ValueError`` is value is not a valid ``str``.

    """
    ### quering ###

    if isinstance(value, str) == False:
        try:
            raise ValueError("Not a string")
        except ValueError:
            print(f"\n\nThe parameter '{param_name}' must be a str, but it's type is '{type(value)}'\n\n")
            raise
        return True




def _check_is_subplots(value, param_name):
    """This function checks if a ``value`` is a ``matplotlib.axes.SubplotBase``.

    This function verifies if the parameter ``value`` is the type of ``matplotlib.axes.SubplotBase`` (``True`` or ``False``). If so, it returns ``True``. If it is not, the function raises a ``ValueError``.

    Parameters
    ----------
    value : any type
        The value to be evaluated.
    param_name : ``str``
        The original name of the parameter passed through the parameter ``value``.

    Returns
    -------
    ``True`` if ``value`` is a ``matplotlib.axes.SubplotBase``
    ``ValueError`` if ``valuew`` is not a ``matplotlib.axes.SubplotBase``


    """

    if isinstance(value, SubplotBase) == False:
        try:
            raise ValueError("Not a subplot instance")
        except ValueError:
            print(f"\n\n The parameter '{param_name}' must be type of 'matplotlib.axes.SubplotBase', but it is '{type(value).__name__}'. \n\n")
            raise

    return True



# def _check_list_of_types(lista, param_name, expected_type):
#     """This function checks if all elements in ``lista`` have type equal to ``expected_type``.

#     Parameters
#     ----------
#     lista :  ``list``
#         The list to be tested
#     param_name : ``str``
#         The name of the parameter passed through the parameter ``lista``.
#     expected_type : ``any``
#         The the type to be checked



#     Returns
#     -------
#     ``True`` if ``lista`` all elements have type equal to ``expected_type``
#     ``TypeError`` otherwise.

#     """

#     if all(isinstance(item, expected_type) for item in lista) == False:
#         try:
#             raise TypeError(f"Not a '{expected_type.__name__}' Error")
#         except TypeError:
#             print(f"\n\nAt least one element contained in '{param_name}' is not of type '{expected_type.__name__}'\n\n")
#             raise
#     else:
#         return True

def _check_list_size(lista, param_name, minimum):
    """This function checks if the size of ``lista`` is equal or higher than ``minimum``.

    Parameters
    ----------
    lista :  ``list``
        The list to be tested
    param_name : ``str``
        The name of the parameter passed through the parameter ``lista``.
    minimum : ``int``
        The lower bound (closed)



#     Returns
#     -------
#     ``True`` if ``lista`` is equal or higher than ``minimum``.
#     ``ValueError`` if ``lista`` is lower than ``minimum``.

#     """

    if len(lista) < minimum:
        try:
            raise ValueError(f"Insufficiently sized array'")
        except ValueError:
            print(f"\n\nThe size of the '{param_name}' list must be at least equal to '{minimum}', but its size is equal to '{len(lista)}'\n\n")
            raise
    else:
        return True

# def _check_matching_sample_size(arr1, param_name1, arr2, param_name2):
#     """This function checks if the size of arr1 is equal to the size of arr2.

#     Parameters
#     ----------
#     arr1 :  ``numpy array``
#         One dimension :doc:`numpy array <numpy:reference/generated/numpy.array>`
#     param_name1 : ``str``
#         The name of the parameter passed through the parameter ``arr1``.
#     arr2 :  ``numpy array``
#         One dimension :doc:`numpy array <numpy:reference/generated/numpy.array>`
#     param_name2 : ``str``
#         The name of the parameter passed through the parameter ``arr2``.



#     Returns
#     -------
#     ``True`` if ``arr1.size == arr2.size``
#     ``ValueError`` if ``arr1.size != arr2.size``

#     """

#     if arr1.size != arr2.size:
#         try:
#             raise ValueError("Array sizes are not equal")
#         except ValueError:
#             print(f"\n\nThe size of '{param_name1}' ({arr1.size}) is different from the size of '{param_name2}' ({arr2.size}), but they must be the same\n\n")
#             raise
#     else:
#         return True



def _check_param_values(value, param_name, param_values):
    """This function checks if a ``value`` is a valid param value.

    Parameters
    ----------
    value : ``int`` or ``float``
        The value to be evaluated
    param_name : ``str``
        The original name of the parameter passed through the parameter ``value``.
    param_values : ``list``
        A list with the possible values for ``param_name``


    Returns
    -------
    ``True`` if ``value`` is in ``param_values``
    ``ValueError`` if ``value`` is not in the range: ``min < value < max``

    """    
    if value not in param_values:
        try:
            raise ValueError("Value not allowed")
        except ValueError:
            print(f"\n\n The value of the parameter '{param_name}' cannot be equal to '{value}'.")
            print("Only the following values are accepted for this parameter:")
            for value in param_values:
                print(f">>> {value}")
            print("\n")
            raise    
    else:
        return True







# def _check_sample_size(arr, param_name, minimum):
#     """This function checks if the size of ``arr`` is equal or higher than ``minimum``.

#     Parameters
#     ----------
#     arr :  ``numpy array``
#         One dimension :doc:`numpy array <numpy:reference/generated/numpy.array>`
#     param_name : ``str``
#         The name of the parameter passed through the parameter ``arr``.
#     minimum : ``int``
#         The lower bound (closed)



#     Returns
#     -------
#     ``True`` if ``arr`` is equal or higher than ``minimum``.
#     ``ValueError`` if ``arr`` is lower than ``minimum``.

#     """

#     if arr.size < minimum:
#         try:
#             raise ValueError(f"Insufficient sample size for parameter '{param_name}'")
#         except ValueError:
#             print(f"\n\nThe minimum size required for '{param_name}' is '{minimum}', but it has size equal to '{arr.size}'\n\n")
#             raise
#     else:
#         return True

# def _convert_to_one_D_numpy(arr, param_name):
#     """This function converts a ``arr`` to and ``numpy array`` of 1 dimension

#     Parameters
#     ----------
#     value : any
#         The value to be converted to 1-dimensional numpy array.
#     param_name : ``str``
#         The name of the parameter

#     Returns
#     -------
#     arr :  ``numpy array``
#         One dimension :doc:`numpy array <numpy:reference/generated/numpy.array>`

#     Notes
#     -----
#     Raises ``TypeError`` if the ``arr`` is not not a non-empty ``1-dimensional numpy array``


#     """
#     arr = np.asarray(arr)
#     if isinstance(arr, np.ndarray) == False:
#         try:
#             raise TypeError(f"Unable to turn '{param_name}' into a numpy array properly")
#         except TypeError:
#             raise
#     elif arr.ndim != 1:

#         try:
#             raise TypeError(f"Unable to turn '{param_name}' into a numpy array properly")
#         except TypeError:
#             print(f"\n\nThe resulting array contains {arr.ndim} dimensions, but must contain 1 dimension\n\n")
#             raise
#     elif arr.size == 0:
#         try:
#             raise ValueError(f"Unable to turn '{param_name}' into a numpy array properly")
#         except ValueError:
#             print("\n\nThe resulting array has size equal to 0\n\n")
#             raise
#     else:
#         return arr



































































































# #
