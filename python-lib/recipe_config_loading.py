from safe_logger import SafeLogger

logger = SafeLogger("Time series preparation plugin")


def check_time_column_parameter(recipe_config, dataset_columns):
    if recipe_config.get("datetime_column") not in dataset_columns:
        raise ValueError("Invalid time column selection: {}".format(recipe_config.get("datetime_column")))


def check_and_get_groupby_columns(recipe_config, dataset_columns):
    long_format = recipe_config.get("advanced_activated", False)
    if long_format:
        groupby_columns = _format_groupby_columns(recipe_config)
        _check_groupby_columns(groupby_columns, dataset_columns)
        return groupby_columns
    else:
        return []


def _format_groupby_columns(recipe_config):
    if recipe_config.get('advanced_activated') and recipe_config.get('groupby_column'):
        logger.warning(
            "The field `Column with identifier` is deprecated. It is now replaced with the field `Time series identifiers`, which allows for several "
            "identifiers. That is why you should preferably use the field 'Time series identifiers'. You can still use 'Column with identifier' if you "
            "have one identifier only")
        groupby_columns = [recipe_config.get('groupby_column')]
    elif recipe_config.get('advanced_activated') and recipe_config.get('groupby_columns'):
        groupby_columns = recipe_config.get('groupby_columns')
    else:
        groupby_columns = []
    return groupby_columns


def _check_groupby_columns(groupby_columns, dataset_columns):
    if len(groupby_columns) == 0:
        raise ValueError("Long format is activated but no time series identifiers have been provided")
    if not all(identifier in dataset_columns for identifier in groupby_columns):
        raise ValueError("Invalid time series identifiers selection: {}".format(groupby_columns))
