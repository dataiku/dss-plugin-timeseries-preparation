# -*- coding: utf-8 -*-
import logging
from dataiku.customrecipe import get_recipe_config
from dku_timeseries import ExtremaExtractor
from commons import check_python_version, get_input_output, get_extrema_extraction_params
from recipe_config_loading import check_time_column_parameter, check_and_get_groupby_columns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='timeseries-preparation plugin %(levelname)s - %(message)s')

check_python_version()

# --- Setup
(input_dataset, output_dataset) = get_input_output()
recipe_config = get_recipe_config()
input_dataset_columns = [column["name"] for column in input_dataset.read_schema()]
check_time_column_parameter(recipe_config, input_dataset_columns)
datetime_column = recipe_config.get('datetime_column')
extrema_column = recipe_config.get('extrema_column')
groupby_columns = check_and_get_groupby_columns(recipe_config, input_dataset_columns)
params = get_extrema_extraction_params(recipe_config)

# --- Run
df = input_dataset.get_dataframe()
extrema_etractor = ExtremaExtractor(params)
output_df = extrema_etractor.compute(df, datetime_column, extrema_column, groupby_columns=groupby_columns)

# --- Write output
output_dataset.write_with_schema(output_df)
