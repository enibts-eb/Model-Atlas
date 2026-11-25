"""General Utilities"""
import os
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

import plotly.io as pio
import plotly.offline
from plotly.basedatatypes import BaseFigure

def _null_coalesce(a,b):
    """Return 'b' if 'a' is None, otherwise 'a'."""
    return b if a is None else a

def _na_coalesce(a,b):
    """Return 'b' if 'a' is None, missing or an empty string, otherwise 'a'."""
    if a is None:
        return b
    
    value = a
    if isinstance(a, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        if len(a) != 1:
            return a
        value = list(a)[0]
    
    if isinstance(value, str):
        return b if value == "" else a
    
    try:
        if pd.isna(value):
            return b
    except (TypeError, ValueError):
        pass
    
    return a

def _is_single_number(value):
    """Return True for a single real number, False for bools, NaNs and anything else."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if not isinstance(value, (Real, np.number)):
        return False
    return not np.isnan(value)

def unitise(data, var, min_val, max_val):
    """Unitise a numeric variable to the range 0 to 1.
    Caps a variable at min_val and max_val then scales it between 0 and 1. Returns a modified copy. The raw data is never mutated.
    
    Parameters
    ----------
    data: pandas.DataFrame
        Source data.
    var: str
        Name of the column to unitise.
    min_val: float
        Lower cap. Values below this are set to 0.
    max_val: float
        Upper cap. Values above this are set to 1.
    
    Returns
    -------
    pandas.DataFrame
        A copy of 'data' with 'var' scaled 0 to 1. Missing values remain missing and non numeric strings become missing.
        
    Raises
    ------
    TypeError
        If 'data' is not a pandas DataFrame or if 'min_val' or 'max_val' is not a single non missing number.
    ValueError
        If 'var' is not a column of 'data' or if 'min_val' is not less than 'max_val'
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
        
    if var not in data.columns:
        raise ValueError(f"var column {var!r} not found in data.")
    
    if not (_is_single_number(min_val) and _is_single_number(max_val)):
        raise TypeError("min_val and max_val must be a single numeric value.")
        
    if min_val >= max_val:
        raise ValueError("min_val must be less than max_val.")
        
    out = data.copy()
    column = out[var]
    
    if isinstance(column.dtype, pd.CategoricalDtype):
        column = column.astype(object)

    x = pd.to_numeric(column, errors =  'coerce').astype(float)
    out[var] = (x.clip(lower = min_val, upper = max_val) - min_val)/(max_val - min_val)     
    
    return out
    
def _is_plot_item(item):
    """Returns True for objects save_plots() can write: plotly figures or HTML strings."""
    if isinstance(item, str):
        return True
    
    #plotly figures share BaseFigure
    return isinstance(item, BaseFigure)

def save_plots(plots, file, selfcontained = True, libdir = None):
    """Save one or more Plotly figures to a single HTML file.
    
    Parameters
    ----------
    plots: plotly figure, str or list/tuple/dict of them
        What to save. One of:
        * a single Plotly figure or a string of HTML,
        * a list or tuple of figures or HTML strings,
        * a dict of them as returned for several datasets
        
        A single object is wrapped into a length-1 list.        
    file: str or path-like
        Path to the output HTML file.
    selfcontained: bool, default True
        Embed the Plotly JavaScript library in the file so it works offline and can be shared on its own. 
    libdir: str or None, default None
        Only used when 'selfcontained = False'. Directory for the Plotly JavaScript file.
        
    Returns
    -------
    str
        The path to the generated HTML file.
    
    Raises
    ------
    TypeError
        If 'file' is not a single file path, if 'plots' is empty or not a figure/HTML string/list/tuple/dict or if any element of 
        'plots' is not a Plotly figure or HTML string.
    """
    if not isinstance(file, (str, os.PathLike)) or (isinstance(file, str) and file == ""):
        raise TypeError("File must be a single file path.")
        
    file = Path(file)
    
    if _is_plot_item(plots):
        items = [plots]
        positions = [0]
    elif isinstance(plots, dict):
        items = list(plots.values())
        positions = list(plots.keys())
    elif isinstance(plots, (list, tuple)):
        items = list(plots)
        positions = list(range(len(items)))
    else:
        items = []
        positions = []
    
    if len(items) == 0:
        raise TypeError("Plots must be a Plotly figure or HTML string or a non-empty list, tuple or dict of them.")

    bad = [position for position, item in zip(positions, items) if not _is_plot_item(item)]
    if bad:
        raise TypeError("Every element of plots must be a Plotly figure or HTML string.\n"
                        f"Element(s) {', '.join(repr(b) for b in bad)} is/are not.")

    if selfcontained:
        head_script = f'<script type = "text/javascript">{plotly.offline.get_plotlyjs()}</script>'
    else:
        if libdir is None:
            libdir = f"{file.stem}_files"
        
        libdir_path = Path(libdir)
        if not libdir_path.is_absolute():
            libdir_path = file.parent/libdir_path
            
        libdir_path.mkdir(parents = True, exist_ok = True)
        js_path = libdir_path/"plotly.min.js"
        if not js_path.exists():
            js_path.write_text(plotly.offline.get_plotlyjs(), encoding = "utf-8")
            
        src = os.path.relpath(js_path, start = file.parent or Path("."))
        head_script = f'<script type="text/javascript" src="{Path(src).as_posix()}"></script>'
        
    body = []
    for item in items:
        if isinstance(item, str):
            body.append(item)
        else:
            body.append(pio.to_html(item, full_html = False, include_plotlyjs = False, default_height = "450px"))
            
    html = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8"/>\n'
        f"{head_script}\n"
        "</head>\n"
        "<body>\n"
        + "\n".join(body)
        + "\n</body>\n"
        "</html>\n"
    )
    
    file.parent.mkdir(parents = True, exist_ok = True)
    file.write_text(html, encoding = "utf-8")

    return str(file)
