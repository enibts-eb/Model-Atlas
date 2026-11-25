# Helper functions for numeric binning
import numpy as np
import pandas as pd


def _assert_col_exists(data, cols, arg_name):
    """Raise an error if any requested columns are missing from the data.
    
    Parameters
    ---------
    data : pandas.DataFrame
        DataFrame containing the expected columns. 
    cols : str or iterable of str
        Column name(s) expected to exist in 'data'
    arg_name: str
        Name of the argument supplying the column names, used in the message
    
    Raises
    ------
    ValueError
        If one or more requested columns are not present in 'data'. 
    
    """
    
    if isinstance(cols, str):
        cols = [cols]
    
    missing_cols = [col for col in dict.fromkeys(cols) if col not in data.columns]
    
    if missing_cols:
        quoted = ", ".join(repr(col) for col in missing_cols)
        raise ValueError(f"{arg_name} column(s) not found in data: {quoted}")
        
def _round_significant(values, digits = 7):
    """Round numeric values to a specified number of significant digits.
    Uses correctly rounded decimal formatting which avoids the precision problems of scaling by power of ten.
    """

    values = np.asarray(values, dtype = float)
    return np.array([float(f"{value:.{digits}g}") for value in values.ravel()]).reshape(values.shape)

def _round(values, decimals = 2):
    """Round numeric values to specific decimals places. Defined in order to achieve precision and avoid mismatches."""
    values = np.asarray(values, dtype = float)
    return np.array([round(float(value), int(decimals)) for value in values.ravel()]).reshape(values.shape)

def _format_break(value, digits):
    """Format the number to a set number of significant digits. This drops trailing zeros and switches to scientific notation for 
    very large or small numbers."""
    value = 0.0 + float(value)
    
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "Inf"
    if np.isneginf(value):
        return "-Inf"
    
    return f"{value:.{digits}g}"

def _cut(x, breaks, include_lowest = False, right = True, dig_lab = 3):
    """
    Parameters
    ----------
    x: array-like
        Numeric values to bin. Missing values give missing bins. 
    breaks : array-like
        Break points. They are sorted and must be unique. 
    include_lowest: bool, default False
        Include values equal to the lowest break (or the highest when right = False) in the first(last) interval. 
    right : bool, default True
        Intervals are closed on the right e.g. (a,b]
    dig_lab: int, default 3
        Significant digits used for the labels. More digits are tried until all labels are distinct. 
    
    Returns
    -------
    pandas.Categorical
        Unordered categorical whose categories are all interval labels in order, including any that end up unused.
    
    Raises
    ------
    ValueError
        If fewer than two break points are given or if the break points are not unique.
    """
    
    x = np.asarray(x, dtype = float)
    breaks = np.sort(np.asarray(breaks, dtype = float).ravel())
    nb = breaks.size
    
    if nb < 2:
        raise ValueError("At least two break points are required.")
    if np.unique(breaks).size != nb:
        raise ValueError("Breaks are not unique.")
        
    # Increase the digits until neighbouring labels differ. 
    ok = False
    for dig in range(dig_lab, max(12, dig_lab) + 1):
        ch_br = [_format_break(value, dig) for value in breaks]
        ok = all(ch_br[i] != ch_br[i+1] for i in range(nb - 1))
        if ok:
            break
        
    if ok:
        left_bracket, right_bracket = ("(", "]") if right else ("[", ")")
        labels = [f"{left_bracket}{ch_br[i]},{ch_br[i+1]}{right_bracket}" for i in range(nb-1)]
        if include_lowest:
            if right:
                labels[0] = "[" + labels[0][1:]
            else:
                labels[-1] = labels[-1][:-1] + "]"
    else:
        labels = [f"Range_{i}" for i in range(1, nb)]
        
    codes = np.full(x.shape, -1, dtype = int)
    valid = ~np.isnan(x)
    xv = x[valid]
    
    if right:
        position = np.searchsorted(breaks, xv, side = "left")
        if include_lowest:
            position= np.where(xv == breaks[0], 1, position)
    else:
        position = np.searchsorted(breaks, xv, side = "right")
        if include_lowest:
            position = np.where(xv == breaks[-1], nb - 1, position)
            
    in_range = (position >= 1) & (position <= nb - 1)
    codes[valid] = np.where(in_range, position - 1, -1)
    
    return pd.Categorical.from_codes(codes, categories = labels)
        
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
        If rounding the break point to 7 significant digits leaves fewer than two distinct break points.
    """
    
    x_sorted = np.asarray(x_sorted, dtype = float)
    n = x_sorted.size
    
    indices = np.unique(np.linspace(n/bins, n, num = bins).astype(int))
    indices = indices[indices >= 1]
    
    breaks = np.concatenate(([x_sorted[0]], x_sorted[indices -1]))
    breaks = _round_significant(breaks, digits = 7)
    breaks = pd.unique(breaks).astype(float)
    
    # Clamp endpoints to the actual data range. 
    breaks[0] = np.min(x_sorted)
    breaks[-1] = np.max(x_sorted)
    
    if breaks.size < 2:
        raise ValueError("At least two distinct break points are required. Rounding to 7 significant digits may have collapsed the break points.")
        
    return _cut(x_sorted, breaks = breaks, include_lowest = True, dig_lab = 7)

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
    ValueError
        If 'x' has no non-missing values, if its minimum or maximum is infinite or if fewer than two distinct break points remain
        after rounding (the data has only one unique value).
    """
    
    values = np.asarray(x, dtype = float)
    
    if np.all(np.isnan(values)):
        raise ValueError("x must contain at least one non-missing value.")
        
    minimum = np.nanmin(values)
    maximum = np.nanmax(values)    
    
    if not (np.isfinite(minimum) and np.isfinite(maximum)):
        raise ValueError("x must contain finite values.")

    spread = maximum - minimum

    if np.isfinite(spread) and spread > 0:
        magnitude = np.floor(np.log10(spread/bins))
    else: 
        magnitude = 0
        
    decimals = int(2 - magnitude) if magnitude < 0 else 0

    breaks= np.linspace(minimum, maximum, num = bins + 1)
    breaks = _round(breaks, decimals = decimals)
    breaks = pd.unique(breaks).astype(float)
    
    # Clamp endpoints to the actual data range. round() can push the lowest break above min(x) or the highest below
    # max(x) which would turn real values into missing bins. 
    breaks[0] = minimum
    breaks[-1] = maximum

    if breaks.size < 2:
        raise ValueError("At least two distinct break points are required. The data may contain only one unique value.")
        
    return _cut(values, breaks = breaks, include_lowest = True, dig_lab = max(7, decimals + 3))

def _remap_sorted_bins(binned_sorted, ord_, n):
    """Place sorted-order bin labels back into the original row positions. 
    
    Parameters
    ---------
    binned_sorted : pandas.Categorical or array-like
        Bin labels corresponding to the non-missing values in ascending order.
    ord_ : array-like of int
        Original zero-based positions of the non-missing values, ordered by ascending value. 
    n : int
        Length of the original vector.
        
    Returns
    -------
    numpy.ndarray
        Object array of length 'n' containing bin labels with missing positions represented by 'None'
    
    """
    
    labels = np.full(n, None, dtype = object)
    sorted_labels = pd.Series(binned_sorted, dtype = object)
    labels[np.asarray(ord_, dtype = int)] = sorted_labels.where(sorted_labels.notna(), None).to_numpy()
    
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
        raise ValueError("type_agg must be either 'equal_range' or 'equal_exposure'.")
    
    non_na_idx = np.flatnonzero(~np.isnan(x))

    # Original positions of non-NA values, ordered by value. 
    ord_ = non_na_idx[np.argsort(x[non_na_idx], kind = 'stable')]
    
    cat_result = bin_fn(x[ord_], bins)
    labels = _remap_sorted_bins(cat_result, ord_, len(x))

    return {
        "labels" : labels,
        "cat" : cat_result
    }
    
