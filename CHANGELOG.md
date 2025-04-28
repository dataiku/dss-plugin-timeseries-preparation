# Changelog


## Version 2.1.1 - Bugfix release - 2025-04
### Resampling recipe
- :bug: support integer columns containing null values
- :bug: support float values in resampling

## Version 2.1.0 - New feature release - 2024-12
### Resampling recipe
- :date: Support selecting custom dates for the start and end of extrapolation

## Version 2.0.5 - Bugfix release - 2025-01
- :bug: Windowing would fail on quarterly and yearly frequencies

## Version 2.0.4 - Bugfix release - 2024-02
- :bug: Plugin resamples on columns when they should be interpreted as categorical columns

## Version 2.0.3 - Bugfix release - 2023-04
- Updated code-env descriptor for DSS 12

## Version 2.0.2 - Bugfix release - 2023-01
- 🪲 Fix the bug that was adding an extra date at the end after resampling when the last input timestamp was exactly at the end of a period (week, month, half-year, year)

## Version 2.0.1 - Bugfix release - 2021-06
- :bug: Keep the empty values rather than filtering them with the extrapolation method "Don't extrapolate (impute nulls)"
- :scissors: Add the extrapolation method "Don't extrapolate (no imputation)" to filter missing values
- :pencil: Edit plugin.json to reflect the changes made in 2.0.0

## Version 2.0.0 - New feature and bugfix release - 2021-05
### New recipe - Time series decomposition
- :chart_with_upwards_trend: Decompose the time series into trend, seasonal and residuals using Seasonal and Trend decomposition using Loess (STL) 
### All the recipes
- Improve long format options
    - :v: Handle multiple identifiers
    - :1234: Allow for numerical columns as time series identifiers
- The plugin no longer supports Python 2.7. Yet, its previous recipes, namely Resampling, Windowing, Interval extraction, and Extrema extraction are still running with a Python 2.7 code env. 
### Resampling recipe
- :date: Add more frequencies: business days, quarters, semi-annual
- :thought_balloon: Impute categorical values during interpolation and extrapolation
- 🪲 Bugfix: prevent the recipe from occasionnaly adding an empty row at the end of the output dataset
### Interval extraction recipe
- 🪲 Bugfix: when we set acceptable deviation = 0, minimal segment duration = 0, the first row belongs to the interval and the second does not, the first row
 is no longer missing from the interval
 
### Windowing recipe
- 🪲 Fix the bug occurring with weekly, monthly and annual time series which failed to convert to offsets
  

## Version 1.0.0 - Initial release - 2019-11
Add visual recipes to prepare time series data
### Resampling recipe
- :chart_with_downwards_trend: Resample time series data 
- :date: Choose frequencies from nanoseconds to years
### Windowing recipe
- :bookmark: Compute aggregations or filter a time series using a rolling window. 
- :left_right_arrow: The window size can vary from nanoseconds to years 
### Extrema extraction recipe
- :mount_fuji: Extract values around an extremum
### Interval extraction recipe
- :scissors: Identify intervals or segments of the time series where the values fall within a given range 

### All the recipes
- :point_up: Handle long format with a unique identifier
- :snake: Support Python 2.7 and Python 3.6

