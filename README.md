# Timeseries Preparation Plugin
The timeseries preparation plugin is a set of python recipes to help you prepare timeseries data.

Sandbox project: http://challenge.dataiku.com:11700/projects/PlugInTest/flow/

## Recipe - Resampling

### Input Dataset:
**The input must have only one timestamp column as it will be automatically detected**.

### Input parameters:
- **Time unit**
- **Time step size**
- **Offset**: starting rows to remove.
- **Crop**: ending rows to remove.
- **Interpolation method**:'nearest', 'previous', 'next', 'linear', 'quadratic', 'cubic', 'barycentric'
- **Extrapolation method**: 'nearest', 'clip' or 'same as interpolation'
- **Column to groupby**: in case in one dataset we have multiple timeserie (one per product/engine/...)
