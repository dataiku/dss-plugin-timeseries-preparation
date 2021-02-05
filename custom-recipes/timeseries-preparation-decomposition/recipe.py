from dataiku.customrecipe import get_recipe_config

from commons import get_input_output
from dku_config.stl_config import STLConfig
from dku_config.classical_config import ClassicalConfig
from dku_timeseries.stl_decomposition import STLDecomposition
from dku_timeseries.classical_decomposition import ClassicalDecomposition
from timeseries_preparation.preparation import TimeseriesPreparator

(input_dataset, output_dataset) = get_input_output()

config = get_recipe_config()
if config.get("time_decomposition_method") == "STL":
    dku_config = STLConfig()
elif config.get("time_decomposition_method") == "classical":
    dku_config = ClassicalConfig()

input_df = input_dataset.get_dataframe()
dku_config.add_parameters(config, input_df)

timeseries_preparator = TimeseriesPreparator(
    time_column_name=dku_config.time_column,
    frequency=dku_config.frequency,
    target_columns_names=dku_config.target_columns,
    timeseries_identifiers_names=dku_config.timeseries_identifiers
)

df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)

if dku_config.time_decomposition_method == "STL":
    decomposition = STLDecomposition(dku_config)
elif dku_config.time_decomposition_method == "classical":
    decomposition = ClassicalDecomposition(dku_config)
transformed_df = decomposition.fit(df_prepared)

transformation_df = output_dataset.write_with_schema(transformed_df)
