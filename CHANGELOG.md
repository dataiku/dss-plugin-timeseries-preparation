# Changelog

## Version 2.0.1 - Bugfix release - 2021-06
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

