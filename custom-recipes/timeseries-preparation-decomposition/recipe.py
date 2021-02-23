from dataiku.customrecipe import get_recipe_config

from commons import get_input_output
from dku_timeseries.dku_decomposition.helpers import check_and_load_params
from timeseries_preparation.preparation import TimeseriesPreparator

(input_dataset, output_dataset) = get_input_output()

config = get_recipe_config()
input_dataset_columns = [column["name"] for column in input_dataset.read_schema()]
(dku_config, input_validator, decomposition) = check_and_load_params(config, input_dataset_columns)

timeseries_preparator = TimeseriesPreparator(dku_config)

input_df = input_dataset.get_dataframe()
df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
input_validator.check(df_prepared)
transformed_df = decomposition.fit(df_prepared)

transformation_df = output_dataset.write_with_schema(transformed_df)
