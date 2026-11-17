import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go

DEFAULT_BAR_COLOUR = "#2563EB"

SPLIT_COLOURS = [
    "#0072B2" #BLUE
   ,"#E69F00" #ORANGE
   ,"#009E73" #GREEN
   ,"#CC79A7" #PURPLE/PINK
   ,"#D55E00" #RED/ORANGE
   ,"#56B4E9" #LIGHT BLUE
   ,"#F0E442" #YELLOW
   ,"#000000" #BLACK
]

def _format_significant(value, digits = 7):
    """Format a numeric value using a specified number of digits."""
    return f"{value:.{digits}g}"

def _smart_level_order(values):
    """Return unique values in their existing categorical order."""
    values = pd.Series(values)
    
    if isinstance(values.dtype, pd.CategoricalDtype):
        present = set(values.dropna())
        return [
            level
            for level in values.cat.categories
            if level in present
        ]
    
    return list(pd.unique(values.dropna()))
    
# Plot without a split

def _plot_distribution_simple(agg, var, colour = DEFAULT_BAR_COLOUR):
    """Plot exposure by target bin without a split variable."""
    agg_df = agg.copy()
    
    x_levels = _smart_level_order(agg_df["_x_bin"])
    total_exposure = agg_df["exposure"].sum()

    if total_exposure > 0:
        percentages = agg_df["exposure"] / total_exposure * 100
    else:
        percentages = pd.Series(
            np.zeros(len(agg_df)),
            index = agg_df.index
        )
        
    hover_text = [
        (
            f"Exposure: {x_value} = "
            f"{_format_significant(exposure,7)} "
            f"({percentage:.1f}%)"
        )    
        for x_value, exposure, percentage in zip(
            agg_df["_x_bin"], 
            agg_df["exposure"],
            percentages
        )
    ]
    
    figure = go.Figure()
    
    figure.add_bar(
        x = agg_df["_x_bin"].astype(str),
        y = agg_df["exposure"],
        name = 'Exposure',
        marker_color = colour,
        hovertext = hover_text,
        hoverinfo = "text"
    )
        
    figure.update_layout(
        xaxis = {
            "title" : var,
            "categoryorder" : "array",
            "categoryarray" : [str(value) for value in x_levels]
        },
        yaxis = {
            "title" : "Exposure",
            "showgrid" : True,
            "autorange" : True
        },
        legend = {
            "x" : 1.02,
            "y" : 0.5
        },
        margin ={
            "t" : 25,
            "b" : 100, 
            "l" : 50,
            "r" : 80
        },
        hovermode = "x",
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rbga(0,0,0,0)"
    )
    
    return figure

# Plot with a split

def _plot_distribution_split(agg, var, split, colours = None):
    """Plot exposure by target bin and split group."""
    
    agg_df = agg.copy()
    
    if colours is None:
        colours = SPLIT_COLOURS
        
    x_levels = _smart_level_order(agg_df["_x_bin"])
    
    groups = sorted(
        group
        for group in agg_df["split"].dropna().unique()
        if group != "__none__"
    )
    
    figure = go.Figure()
    
    for index, group in enumerate(groups):
        subset = agg_df.loc[agg_df["split"] == group].copy()
        group_total = subset["exposure"].sum()
        
        if group_total > 0:
            percentages = subset["exposure"] / group_total * 100
        else: 
            percentages = pd.Series(
                np.zeros(len(subset)),
                index = subset.index
            )
        
        hover_text = [
            (
                f"Exposure ({group}):"
                f"{_format_significant(exposure, 7)}"
                f"({percentage:.1f}% of {group})"
            )
            for exposure, percentage in zip(
                    subset["exposure"],
                    percentages)
            
        ]
        
        figure.add_bar(
            x = subset["_x_bin"].astype(str),
            y = subset["exposure"],
            name = str(group),
            marker_color = colours[index % len(colours)],
            hovertext = hover_text,
            hoverinfo = "text"
        )
        
        figure.update_layout(
            barmode = "group",
            xaxis = {
                "title" : var, 
                "categoryorder" : "array",
                "categoryarray" : [str(value) for value in x_levels]
            },
            yaxis = {
                "title" : "Exposure",
                "showgrid" : True, 
                "autorange" : True
            },
            legend = {
                "x" : 1.02,
                "y" :0.5,
                "title" : {
                    "text" : f"<b>{split}</b>"
                }
            },
            margin = {
                "t" : 25,
                "b" : 100,
                "l" : 50, 
                "r" : 80
            },
            hovermode = "x",
            plot_bgcolor = "rgba(0,0,0,0)",
            paper_bgcolor = "rbga(0,0,0,0)"
        )
 
# Bin creation helper

def _make_distribution_bins(values, bins, type_agg, time_unit = None):
    """Create aligned target labels for distribution aggregation."""
    series = values.copy()
    non_missing = series.dropna()

    result = pd.Series(pd.NA, index = series.index, dtype = "object")

    if non_missing.empty: 
        return result
    
    if pd.api.types.is_datetime64_any_dtype(non_missing):
        if time_unit:
            result.loc[non_missing.index] = (non_missing.dt.to_period(time_unit).astype(str))
        else:
            result.loc[non_missing.index] = non_missing.astype(str)
            
        return result
            
    if pd.api.types.is_numeric_dtype(non_missing):
        if non_missing.nunique() > bins:
            binned = _bin_numeric(
                non_missing.to_numpy(),
                bins = bins,
                type_agg = type_agg
            )
            
            result.loc[non_missing.index] = binned["labels"]
        else:
            result.loc[non_missing.index] = non_missing.astype(str)
            
        return result
    
    result.loc[non_missing.index] = non_missing.astype(str)
    
    return result        
        
def distribution(
    data,
    var = "target",
    exposure = "exposure",
    split = None,
    bins = 35, 
    time_unit = None,
    type_agg = "equal_exposure",
    ret = "plot",
    verbose = False,
    colour = DEFAULT_BAR_COLOUR,
    split_colours = None            
):
    """ Plot or return an exposure-weighted distribution of a target variable. 
    
    Parameters
    ----------
    data : pandas.DataFrame
        Source data.
    var : str, default = "target"
        Target column whose distribution will be calculated.
    exposure: str, default = "exposure"
        Exposur-weight column. If it is absent, each row receives a weight of one. 
    split : str or None, default = None
        Optional column used to create grouped bars.
    bins : int, default = 35
        Number of bins for numeric targets with more unique values than bins.
    time_unit: str or None, default = None
        Frequency used to group datetime targets, such as 'D' (Day), 'W' (Week), 'M' (Month) or 'Y' (Year)
    type_agg : {"equal_exposure", "equal_range"}
        Numeric binning strategy.
    ret: {"plot", "data"}
        Return a Plotly figure or the aggregaed DataFrame
    verbose : bool, default = False
        Report how many missing target values were moved to an 'NA' category. 
    colour : str, default = '#2563EB'
        CSS colour used when no split column is supplied. 
    split_colours : sequence of str or None
        Colours used for the split group.
        
    Returns
    -------
    plotly.graph_objects.Figure or pandas.DataFrame or None
        Plot or aggregated data, depending on 'ret' value. """
        
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    
    if var not in data.columns:
        raise ValueError(f"{var!r} column not found in data.")
    
    if split is not None and split not in data.columns:
        raise ValueError(f"{split!r} column not found in data.")
        
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer.")
        
    valid_types = {"equal_exposure", "equal_range"}
    
    if type_agg not in valid_types:
        raise ValueError("type_agg must be 'equal_exposure' oe 'equal_range'.")
        
    if ret not in {"plot", "data"}:
        raise ValueError("ret must be a 'plot' or 'data'.")
        
    columns = [var]
    
    if exposure in data.columns:
       columns.append(exposure)
       
    if split is not None:
        columns.append(split)
    
    frame = data.loc[:, list(dict.fromkeys(columns))].copy()
    
    exposure_column = exposure
    
    if exposure not in frame.columns:
        exposure_column = "_exposure"
        frame[exposure_column] = 1.0
        
    frame[exposure_column] = pd.to_numeric(frame[exposure_column], errors = "coerce")
        
    if frame[exposure_column].isna().any():
        raise ValueError(f"{exposure!r} must contain only numeric, non missing values.")
        
    non_missing = frame[var].dropna()
    unique_count = non_missing.nunique()
    
    is_numeric = pd.api.types.is_numeric_dtype(frame[var])
    is_datetime = pd.api.types.is_datetime64_any_dtype(frame[var])

    if (unique_count > 2000 and not is_numeric and not (is_datetime and time_unit)):
        warnings.warn(f"{var!r} has {unique_count:,} unique values" "(maximum 2,000 for a non numeric target. Skipping.", stacklevel = 2)        
        return None
    
    missing_count = int(frame[var].isna().sum())
    
    if verbose and missing_count > 0:
        print(f"{missing_count:,} missing {var!r} values(s)" "were moved to the trailing 'NA' category.")
        
    frame["_x_bin"] = _make_distribution_bins(frame[var], bins = bins, type_agg = type_agg, time_unit = time_unit)
    frame["_x_bin"] = frame["_x_bin"].astype(object)
    frame.loc[frame[var].isna(), "_x_bin"] = "NA"
    frame["_x_bin"] = frame["_x_bin"].astype(str)
    
    group_columns = ["_x_bin"]
    
    if split is not None:
        frame["split"] = frame[split].astype(object)
        frame["split"] = frame["split"].where(frame["split"].notna(), "NA")
        frame["split"] = frame["split"].astype(str)
        group_columns.append("split")
    else: 
        frame["split"] = "__none__"

    aggregated = (
        frame.groupby(
            group_columns,
            observed = True, 
            sort = False, 
            dropna = False
        )[exposure_column].sum().reset_index(name = "exposure")
    )            
     
    if split is None:
        aggregated["split"] = "__none__"
        
    if ret == "data":
        return aggregated.rename(columns = {"_x_bin" : var})
        
    has_split = (split is not None and aggregated["split"].nunique() > 1)
    
    if has_split:
        return _plot_distribution_split(aggregated, var = var, split = split, colours = split_colours)
    
    return _plot_distribution_simple(aggregated, var = var, colour = colour)


    