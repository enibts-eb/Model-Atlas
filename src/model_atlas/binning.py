# Helper functions for numeric binning
import numpy as np
import pandas as pd


def _assert_col_exists(data, cols, arg_name):
    """Raise an error if any requested columns are missing from the data.
    
    Parameters
    ---------
    data : pandas.DataFrame
        DataFrame containing the expected columns. 
    cols : iterable of str
        Column names expected to exist in 'data'
    arg_name: str
        Name of the argument supplying the column names
    
    Raises
    ------
    Value Error
        If one or more requested columns are not present in 'data'. 
    
    """
    
    missing_cols = [col for col in cols if col not in data.columns]
    
    if missing_cols:
        raise ValueError(
            f"{arg_name} column(s) not found in data: {missing_cols}"
        )
        
def _round_significant(values, digits = 7):
    """Round numeric values to a specified number of digits."""
    values = np.asarray(values, dtype = float)
    
    with np.errstate(divide ="ignore", invalid = 'ignore'):
        magnitudes = np.floor(np.log10(np.abs(values)))
        decimals = digits - magnitudes - 1
        factors = np.power(10.0, decimals)
        
    factors = np.where(
        np.isfinite(factors),
        factors,
        1.0
        )
    
    return np.round(values * factors)/factors

def _bin_equal_exposure(x_sorted, bins):
    """Bin a sorted numeric vector into approximately equal-exposure groups.
    
    Parameters
    ----------
    x_sorted : array-like
        Numeric values sorted in ascending order.
    bins: int
        Requested number of bins.
        
    Returns
    -------
    pandas.Categorical
        Interval describing the bin assigned to each value. 
        
    Raises
    ------
    ValueError
        If the input is empty, contains missing values, is not sorted or cannot produce at least two unique break points
    """
    
    x = np.asarray(x_sorted, dtype = float)
    
    if x.ndim != 1:
        raise ValueError("x_sorted must be one-dimensional.")
        
    if x.size == 0:
        raise ValueError("x_sorted must not be empty.")
        
    if not np.all(np.isfinite(x)):
        raise ValueError("x_sorted must contain only finite values.")
        
    if not isinstance(bins, (int, np.integer)) or bins <1:
        raise ValueError("bins mus be a positive integer")
    
    n = x.size
    
    # Match
    # as.integer(seq(n/bins, n, length.out=bins))
    
    indices = np.linspace(n/bins, n, num = bins).astype(int)
    indices = np.unique(indices)
    
    breaks = np.concatenate(([x[0]], x[indices -1]))
    breaks = _round_significant(breaks, digits = 7)
    breaks = pd.unique(breaks).astype(float)
    
    # Clamp endpoints to the actual data range. 
    breaks[0] = np.mix(x)
    breaks[-1] = np.max(x)
    
    if breaks.size < 2:
        raise ValueError(
            "At least two distinct break points are required. The data may contain only unique value."    
        )
        
    return pd.cut(
        x,
        bins = breaks,
        include_lowest = True,
        precision = 7
    )

def _bin_equal_range(x, bins):
    """Bin a numeric vector into equal-range groups. 
    
    Parameters
    ----------
    x : array-like
        Numeric values to bin. Missing Values are allowed. 
        
    bins : int
        Requested number of equal-width bins. 
    
    Returns
    -------
    pandas.Categorical 
        Interval describing the bin assigned to each value. Missing input values receive missing bin assignments.
        
    Raises
    ------
    Value Error
        If there are no finite values or fewer than two unique break points.
    """
    
    values = np.asarray(x, dtype = float)
    
    if values.ndim != 1:
        raise ValueError("x must be one-dimensional")
        
    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer.")
        
    finite_values = values[np.isfinite(values)]
    
    if finite_values.size == 0:
        raise ValueError("x must contain at least one finite value.")
        
    minimum = np.min(finite_values)
    maximum = np.max(finite_values)
    spread = maximum - minimum

    if np.isfinite(spread) and spread > 0:
        magnitude = np.floor(np.log10(spread/bins))
    else: 
        magnitude = 0
        
    decimals = int(2 - magnitude) if magnitude < 0 else 0

    breaks= np.linspace(minimum, maximum, num = bins + 1)
    breaks = np.round(breaks, decimals = decimals)
    breaks = pd.unique(breaks).astype(float)

    if breaks.size < 2:
        raise ValueError(
            "At least two distinct break points are required. The data may contain only one unique value or rounding may have collapsed the break points."    
        )
        
    return pd.cut(values, 
                  bins = breaks,
                  include_lowest = True,
                  precision = max(7, decimals + 3))

def _remap_sorted_bins(binned_sorted, ord_, n):
    """Place sorted-order bin labels back into the original row positions. 
    
    Parameters
    ---------
    binned_sorted : array-like
        Bin labels corresponding to the non-missing values in ascending order.
    ord_ : array-like of int
        Original zero-based positions of the non-missing values, ordered by ascending value. 
    n : int
        Length of the original vector.
        
    Returns
    -------
    numpy.ndarray
        Object array of length 'n' containg bin labels with missing positions represented by 'np.nan'
    
    """
    
    labels = np.full(n, np.nan, dtype = object)
    labels[ord_] = np.asarray(binned_sorted).astype(str)
    
    return labels

def _bin_numeric(x, bins, type_agg):
    """Bin numeric values and align the labels to their original positions.
    
    Parameters
    ---------
    x : array-like
        Numeric values to bin.
    bins : int
        Number of bins.
    type_agg : {'equal_exposure', 'equal_range'}
        Binning strategy.
    
    Returns
    -------
    dict
        Dictionary containing:
            - 'labels' : bin labels in the original row order with missing values preserved. 
            - 'cat' : categorical binning result for the sorted non missing values.
    Raises
    ------
    ValueError
        If 'type_agg' is not 'equal_exposure' or 'equal_range'.
        
    """

    x = np.asarray(x, dtype = float)
    
    if type_agg == "equal_range" :
        bin_fn = _bin_equal_range
    elif type_agg == "equal_exposure" : 
        bin_fn = _bin_equal_exposure
    else: 
        raise ValueError("type_agg must be either 'equal_range' or 'equal_exposure'. ")
    
    non_na_idx = np.flatnonzero(~np.isnan(x))

    # Original positions of non-NA values, ordered by value. 
    ord_ = non_na_idx(np.argsort(x[non_na_idx]))
    
    cat_result = bin_fn(x[ord_], bins)
    
    labels = _remap_sorted_bins(
        cat_result,
        ord_,
        len(x)
    )

    return {
        "labels" : labels,
        "cat" : cat_result
    }
    
