# """This module concentrates the help functions to run the functions behind the scenes

# """


# # Function list:
# #
# #     - LanguageManagment
# #         - get_language(self)
# #         - set_language(self, language)
# #         - __str__(self)
# #         - __repr__(self)
# #
# #     - AlphaManagement(LanguageManagment)
# #         - get_alfa(self)
# #         - set_alfa(self, alfa)
# #         - __str__(self)
# #         - __repr__(self)
# #
# #
# #     - NDigitsManagement(LanguageManagment)
# #         - get_n_digits(self)
# #         - set_n_digits(self, n_digits)
# #         - __str__(self)
# #         - __repr__(self)

# #     - _change_decimal_separator_x_axis(fig, axes, decimal_separator)
# #     - _change_locale(language, decimal_separator=".", local="pt_BR")
# #     - _change_locale_back_to_default(default_locale)
# #     - _check_blank_space(value, param_name, language)
# #     - _check_conflicting_filename(file_name, extension, language)
# #     - _check_decimal_separator(decimal_separator, language)
# #     - _check_figure_extension(value, param_name, language)
# #     - _check_file_exists(file_name)
# #     - _check_file_name_is_str(file_name, language)
# #     - _check_forbidden_character(value, param_name, language)
# #     - _check_plot_design(plot_design, param_name, plot_design_default, plot_design_example, language)
# #     - _check_which_density_gaussian_kernal_plot(which, language)
# #     - _export_to_csv(df, file_name="my_data", sep=',', language)
# #     - _export_to_xlsx(df_list, language, file_name=None, sheet_names=[None,None])
# #     - _flat_list_of_lists(my_list, param_name, language)
# #     - _raises_when_fit_was_not_applied(func_name, language, name)
# #     - _replace_last_occurrence(value, old, new, occurrence)
# #     - _sep_checker(sep, language)
# #     - _truncate(value, language, decs=None)



# #########################################
# ################ Imports ################
# #########################################

# ###### Standard ######
# import locale
# import logging
# from pathlib import Path
# import traceback

# ###### Third part ######
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# ###### Home made ######

# from pygnosis.utils import checkers


# ###########################################
# ################ Functions ################
# ###########################################





# class AlphaManagement():
#     """Instanciates a class for ``alpha`` managment. This class is primarily for internal use.

#     """

#     def __init__(self, alpha=None, **kwargs):
#         super().__init__(**kwargs)
#         """Constructs the significance level value

#         Parameters
#         ----------
#         alpha : ``float``
#             The significance level (default is ``None``, which means ``0.05``)

#         Notes
#         -----
#         This method only allows input of type ``float`` and between ``0.0`` and ``1.0``.

#         """

#         if alpha is None:
#             self.alpha = 0.05
#         else:
#             checkers._check_is_float(alpha, "alpha")
#             checkers._check_data_in_range(alpha, "alpha", 0.0, 1.0)
#             self.alpha = alpha

#     def get_alpha(self):
#         """Returns the current ``alpha`` value
#         """
#         return self.alpha

#     def set_alpha(self, alpha):
#         """Changes the ``alpha`` value

#         Parameters
#         ----------
#         alpha : ``float``
#             The new significance level

#         Notes
#         -----
#         This method only allows input of type ``float`` and between ``0.0`` and ``1.0``.

#         """
#         checkers._check_is_float(alpha, "alpha")
#         checkers._check_data_in_range(alpha, "alpha", 0.0, 1.0)
#         self.alpha = alpha

#     def __repr__(self):
#         return f"{self.alpha}"

#     def __str__(self):
#         return f"The current significance level is '{self.alpha}'"









# def _export_to_xlsx(df_list, file_name=None, sheet_names=[None,None], verbose=True):
#     """Export the data to .xlsx file
#     This function is just a wraper around DataFrame.to_excel [1]_ to export excel files.

#     Parameters
#     ----------
#     df_list : list with pandas.DataFrame
#         A list where each element is a DataFrame that will be inserted in a different sheet. The order of DataFrames in this list must match the list of names passed through the sheet_names parameter.
#     file_name : string
#         The name of the file to be exported, without its extension (default = 'my_data')
#     sheetet_names : list (default [None, None])
#         A list where each element is a string that will be used for each sheet. The order of strings in this list must match the list of DataFrames passed through the df_list parameter.



#     Returns
#     -------
#     True if the file is exported
#     ValueError if it was not possible to export the file

#     References
#     ----------
#     .. [1] https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html

#     """
#     checkers._check_is_list(df_list, "df_list")
#     for df in df_list:
#         checkers._check_is_data_frame(df, "df")

#     if file_name is None:
#         file_name = "my_data"
#     else:
#         checkers._check_is_str(file_name, "file_name")
#         checkers._check_forbidden_character(file_name, "file_name")

#     checkers._check_is_list(sheet_names, "sheet_names")
#     for name in sheet_names:
#         checkers._check_is_str(name, "sheet_names")


#     # combando nome corretamente
#     file_name = file_name + ".xlsx"

#     # verificando se os dataframes são validos
#     for df in df_list:
#         checkers._check_is_data_frame(df, "df_list")

#     ### verificando se o arquivo .xlsx já existe
#     if _check_file_exists(file_name):
#         ## caso exista, obter uma lista com o nome das abas do arquivo
#         # obter os sheet names
#         arquivo = pd.ExcelFile(file_name, engine="openpyxl")
#         # iterar esta lista comparando se os sheet_names já existem nela
#         sheet_alredy_exists = [] # lista vazia para acumular nomes repetidos
#         for sheet in arquivo.sheet_names: # olhando nome por nome dentro de sheet_names
#             if sheet in sheet_names:
#                 sheet_alredy_exists.append(sheet) # apendadndo
#             else:
#                 pass
#         # caso pelo menos 1 aba já exista, avisar que o nome escolhido será alterado
#         if len(sheet_alredy_exists) > 0:
#             if verbose:
#                 lista_sheet_names = [f"    --->    '{sheet}'" for sheet in arquivo.sheet_names]
#                 sheet_alredy_exists = [f"    --->    '{sheet}'" for sheet in sheet_alredy_exists]
#                 print(f"The '{file_name}' file contains sheets with the following names:")
#                 for sheet in lista_sheet_names:
#                     print(f"- {sheet}")
#                 print(f"And it was requested to save the data on the following sheets:")
#                 for sheet in sheet_alredy_exists:
#                     print(f"- {sheet}")
#                 print("In order to avoid information loss, new names will be used for the sheets with conflicting sheet name.")
#                 print("No changes will be made to the data contained in these worksheets.")
#             arquivo.close()
#         else:
#             pass # é pass pois caso o sheet name já exista, é apenas para avisar que o nome será alterado. É apenas um aviso
#         # caso não tenha nenhum nome conflitante, inserir novas abas no arquivo fornecido
#         try:
#             with pd.ExcelWriter(file_name, mode="a", if_sheet_exists="new") as writer:
#                 for i in range(len(df_list)):
#                     df_list[i].to_excel(writer, sheet_name=sheet_names[i], index=False, engine="openpyxl")
#             if verbose:
#                 print(f"The data has been exported to the '{file_name}' file")
#         except PermissionError:
#             print(f"You are not current allowed to change the '{file_name}' file")
#             print(f"Hint ---> Maybe the '{file_name}' file is open! Please close the file and try again!")
#             raise
#         except FileNotFoundError: # acredito que o problema com subfolder é resolvido com a proibição do /
#             # logging.error(traceback.format_exc())
#             print(f"You are not current allowed to change the '{file_name}' file")
#             print("Hint ---> If you are creating a file in a subfolder, create the folder in advance!")
#             raise
#     else:
#         ## caso não exista, criar o arquivo e exportar
#         try:
#             with pd.ExcelWriter(file_name) as writer:
#                 for i in range(len(df_list)):
#                     df_list[i].to_excel(writer, sheet_name=sheet_names[i], index=False, engine="openpyxl")
#             if verbose:
#                 print(f"The data has been exported to the '{file_name}' file")
#         except PermissionError:
#             print(f"You are not current allowed to change the '{file_name}' file")
#             print(f"Hint ---> Maybe the '{file_name}' file is open! Please close the file and try again!")
#             raise
#         except FileNotFoundError: # acredito que o problema com subfolder é resolvido com a proibição do /
#             # logging.error(traceback.format_exc())
#             print(f"You are not current allowed to change the '{file_name}' file")
#             print("Hint ---> If you are creating a file in a subfolder, create the folder in advance!")
#             raise
#     return True





# # with tests, with no text with no database, with docstring
# def _check_file_exists(file_name):
#     """This function checks if a file already exists on the current folder

#     Parameters
#     ----------
#     file_name : string
#         The file name (with extension).

#     Return
#     ------
#     True if file already exists
#     or
#     False if file does not exists

#     """
#     file = Path(file_name)
#     if file.exists():
#         return True
#     else:
#         return False




# def _flat_list_of_lists(my_list, param_name):
#     """This function flats a list of lists

#     Parameters
#     ----------
#     my_list : list of lists
#         The list with lists to be flattened. All inner elements must be a list

#     param_name : string
#         The original name of the parameter passed through the parameter 'my_list'.


#     Returns
#     -------
#     A flattened list
#     ValueError if 'my_list' does not contain lists in all its elements.

#     """
#     checkers._check_is_list(my_list, param_name)
#     if all(isinstance(element, list) for element in my_list) == False:
#         try:
#             raise ValueError("Error: not a list of lists")
#         except ValueError:
#             print(f"At least one element of the '{param_name}' list is not a list")
#             raise
#     return [item for sublist in my_list for item in sublist]



# # Death, death, death, death, death, death, death, death, death, Death, death, death https://youtu.be/jRc9dbgiBPI?t=334
