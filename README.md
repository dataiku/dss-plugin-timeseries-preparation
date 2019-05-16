# Timeseries Preparation Plugin
The timeseries preparation plugin is a set of python recipes to help you prepare timeseries data.

## Recipe - Resampling

### Input Dataset:
**The input must have only one timestamp column as it will be automatically detected**.

### Input parameters:
- **Time unit**
- **Time step size**
- **Offset**: numbe of starting rows to remove.
- **Crop**: number of ending rows to remove.
- **Interpolation method**:'nearest', 'previous', 'next', 'linear', 'quadratic', 'cubic', 'barycentric'
- **Extrapolation method**: 'nearest', 'clip' or 'same as interpolation'
- **Column to groupby**: in case in one dataset we have multiple timeserie (one per product/engine/...)
