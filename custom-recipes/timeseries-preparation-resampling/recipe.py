# -*- coding: utf-8 -*-

from dataiku.customrecipe import get_recipe_config

from dku_timeseries import Resampler
from io_utils import get_input_output
from recipe_config_loading import check_and_get_groupby_columns, check_time_column_parameter, check_python_version, get_resampling_params

import inspect

check_python_version()

# --- Setup
(input_dataset, output_dataset) = get_input_output()
recipe_config = get_recipe_config()
schema = input_dataset.read_schema()
input_dataset_columns = [column["name"] for column in schema]
check_time_column_parameter(recipe_config, input_dataset_columns)
groupby_columns = check_and_get_groupby_columns(recipe_config, input_dataset_columns)
datetime_column = recipe_config.get('datetime_column')
params = get_resampling_params(recipe_config)

# use_nullable_integers is only available in DSS >= 13.1
# Prior to this, the plugin does not support integer columns with NaN values
signature = inspect.signature(input_dataset.get_dataframe)
can_use_nullable_integers = "use_nullable_integers" in signature.parameters

if can_use_nullable_integers:
    df = input_dataset.get_dataframe(infer_with_pandas=False, use_nullable_integers=True, bool_as_str=True)
else:
    df = input_dataset.get_dataframe(infer_with_pandas=False)

resampler = Resampler(params)
output_df = resampler.transform(df, datetime_column, groupby_columns=groupby_columns)

# int columns must be resampled into int values
dss_expected_integer_columns = [
    column["name"]
    for column in schema
    if column["type"] in ["tinyint", "smallint", "int", "bigint"]
]
columns_to_round = [c for c in df.select_dtypes(include="inexact").columns if c in dss_expected_integer_columns]
output_df[columns_to_round] = output_df[columns_to_round].round()


# --- Write output
output_dataset.write_schema(schema)
output_dataset.write_dataframe(output_df)
