# Changelog

## Version 2.0.0 - New feature and bugfix release - 2021-05
### New recipe - Time series decomposition
- Decompose the time series into trend, seasonal and residuals using Seasonal and Trend decomposition using Loess (STL) 
### All the recipes
- Improve long format options
    - Handle multiple identifiers
    - Allow for numerical columns as time series identifiers
### Resampling recipe
- Add more frequencies in the settings : business days, quarters, semi-annual
- Impute categorical values during interpolation and extrapolation
- Bugfix: prevent the recipe from occasionnaly adding an empty row at the end of the output dataset
### Interval extraction recipe
- Bugfix: when we set acceptable deviation = 0, minimal segment duration = 0, the first row belongs to the interval and the second does not, the first row
 is no longer missing from the interval
- Improve wording
 
### Windowing recipe
- Fix the bug occurring with weekly, monthly and annual time series which failed to convert to offsets
  


