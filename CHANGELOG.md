# Changelog

## Version 2.0.0 - New feature and bugfix release - 2021-05
### New recipe - Time series decomposition
- :chart_with_upwards_trend: Decompose the time series into trend, seasonal and residuals using Seasonal and Trend decomposition using Loess (STL) 
### All the recipes
- Improve long format options
    - :v: Handle multiple identifiers
    - :1234: Allow for numerical columns as time series identifiers
- The plugin no longer supports Python 2.7. Yet, its previous recipes, namely Resampling, Windowing, Interval extraction and Extrema extraction, are still running with a Python 2.7 code env. 
### Resampling recipe
- :date: Add more frequencies: business days, quarters, semi-annual
- :thought_balloon: Impute categorical values during interpolation and extrapolation
- 🪲 Bugfix: prevent the recipe from occasionnaly adding an empty row at the end of the output dataset
### Interval extraction recipe
- 🪲 Bugfix: when we set acceptable deviation = 0, minimal segment duration = 0, the first row belongs to the interval and the second does not, the first row
 is no longer missing from the interval
 
### Windowing recipe
- 🪲 Fix the bug occurring with weekly, monthly and annual time series which failed to convert to offsets
  


