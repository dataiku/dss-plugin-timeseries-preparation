import sys

from dku_config.utils import PluginCodeEnvError

if sys.version_info.major == 2:
    raise PluginCodeEnvError("This custom recipe requires a Python 3.6 code env. You are using Python {}.{}. Please create a new Python 3.6 code "
                             "environment for this plugin if you want to use the decomposition recipe".format(sys.version_info.major, sys.version_info.minor))

from time import perf_counter

from dataiku.customrecipe import get_recipe_config

from io_utils import get_input_output, set_column_description
from recipe_config_loading import get_decomposition_params
from safe_logger import SafeLogger
from timeseries_preparation.preparation import TimeseriesPreparator

logger = SafeLogger("Timeseries preparation plugin")

(input_dataset, output_dataset) = get_input_output()
config = get_recipe_config()
input_dataset_columns = [column["name"] for column in input_dataset.read_schema()]
(dku_config, input_validator, decomposition) = get_decomposition_params(config, input_dataset_columns)

timeseries_preparator = TimeseriesPreparator(dku_config)
input_df = input_dataset.get_dataframe(infer_with_pandas=False)
df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
input_validator.check(df_prepared)

start = perf_counter()
logger.info("Decomposing time series...")
transformed_df = decomposition.fit(df_prepared)
logger.info("Decomposing time series: Done in {:.2f} seconds".format(perf_counter() - start))
transformation_df = output_dataset.write_with_schema(transformed_df)
set_column_description(output_dataset, decomposition.columns_descriptions, input_dataset)
