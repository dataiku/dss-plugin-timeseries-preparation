from dataiku.customrecipe import get_recipe_config

from Constants import Method
from commons import get_input_output
from dku_config.stl_config import STLConfig
from dku_config.classical_config import ClassicalConfig
from dku_input_validator.classical_input_validator import ClassicalInputValidator
from dku_input_validator.decomposition_input_validator import DecompositionInputValidator
from dku_timeseries.stl_decomposition import STLDecomposition
from dku_timeseries.classical_decomposition import ClassicalDecomposition
from timeseries_preparation.preparation import TimeseriesPreparator

(input_dataset, output_dataset) = get_input_output()

config = get_recipe_config()
input_dataset_columns = [column["name"] for column in input_dataset.read_schema()]
decomposition_method = Method(config.get("time_decomposition_method"))

if decomposition_method == Method.STL:
    dku_config = STLConfig()
    dku_config.add_parameters(config, input_dataset_columns)
    input_validator = DecompositionInputValidator(dku_config)
elif decomposition_method == Method.CLASSICAL:
    dku_config = ClassicalConfig()
    dku_config.add_parameters(config, input_dataset_columns)
    input_validator = ClassicalInputValidator(dku_config)

timeseries_preparator = TimeseriesPreparator(
    time_column_name=dku_config.time_column,
    frequency=dku_config.frequency,
    target_columns_names=dku_config.target_columns,
    timeseries_identifiers_names=dku_config.timeseries_identifiers
)

input_df = input_dataset.get_dataframe()
df_prepared = timeseries_preparator.prepare_timeseries_dataframe(input_df)
input_validator.check(df_prepared)

if decomposition_method == Method.STL:
    decomposition = STLDecomposition(dku_config)
elif decomposition_method == Method.CLASSICAL:
    decomposition = ClassicalDecomposition(dku_config)
transformed_df = decomposition.fit(df_prepared)

transformation_df = output_dataset.write_with_schema(transformed_df)
