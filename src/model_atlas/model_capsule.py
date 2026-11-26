"""ModelCapsule is a model agnostic container for a fitted model and its pipeline. 
Holds a fitted model together with its data splits, pipeline functions(pre-processing,
feature engineering, post-processing and metadata. Instances are immutable - helper functions
have been defined below to get a new altered copy of a ModelCapsule."""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .utils import _na_coalesce

def _identity(df):
    """Default pre-processing/feature engeering: return 'df' unchanged."""
    return df

def _identity_post(preds, df_raw):
    """Default post-processing: return the predictions unchanged."""
    return preds

def _as_name_tuple(value):
    """Normalise a character vector to a tuple of names.
    A single string becomes a 1-tuple. None becomes '()'. Lists/arrays/series become tuples.
    Anything else is returned unchanged so validation can report it.
    """
    if value is None:
        return ()
    if isinstance(value,str):
        return (value,)
    if isinstance(value, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        return tuple(value)
    return value

def _call_pipeline_fun(fun, fun_name, *args):
    """Call a pipeline function with a helpful error message.
    Catches 'NameError' and re-raises it with advice on how to fix it. Any other error is raised unchanged.
    
    Parameters
    ----------
    fun: callable
        The pipeline function to call.
    fun_name: str
        Name of the pipeline functions used in the error message.
    *args
        Arguments passed on to 'fun'.
        
    Returns
    -------
    Any
        Whatever 'fun' returns.
    
    Raises
    ------
    NameError
        If 'fun' refers to a name that is not defined.
    """
    try:
        return fun(*args)
    except NameError as error:
        raise NameError(f"{fun_name} failed: {error}\n"
                        f"A package used inside {fun_name} is not imported.") from error
    
def _is_h2o_model(model):
    """True if 'model' is an H2O model"""
    return type(model).__module__.split(".")[0] == "h2o"
    
@dataclass(frozen = True)
class ModelCapsule:
    """Model-agnostic container for a fitted model and its pipeline."""
    
    model: Any
    
    train: pd.DataFrame | None = None
    test: pd.DataFrame | None = None
    validation: pd.DataFrame | None = None
    
    feat_eng_fun: Callable = field(default = _identity)
    pre_process_fun: Callable = field(default = _identity)
    post_process_fun: Callable = field(default = _identity)
    
    raw_variables: tuple = (None,)
    x_variables: tuple = (None,)
    
    target_name: str | None = None
    preds_name: str | None = None
    
    exposure_name: str | None = "exposure"
    expo_val: float = 1
    expo_0_rep: float = 0.1
    
    offset_name: str | None = None
    offset_value: float | None = None
    
    model_name: str | None = None
    model_notes: str | None = None
    
    def __post_init__(self):
        self._validate()
    
    # Components check/validation    
    def _validate(self):
        errors = []
        
        # Model
        if self.model is None:
            errors.append("model must supply a fitted model object.")
        
        # Datasets
        for name in ("train", "test", "validation"):
            value = getattr(self, name)
            
            if value is not None and not isinstance(value, pd.DataFrame):
                errors.append(f"{name} must be a pandas DataFrame or None.")
         
        # Pre/Post & Feature Engineering functions
        for name in ("pre_process_fun", "post_process_fun", "feat_eng_fun"):
            if not callable(getattr(self,name)):
                errors.append(f"{name} must be a function.")
         
        # String properties
        string_properties = (
             "target_name"
            ,"preds_name"
            ,"exposure_name"
            ,"offset_name"
            ,"model_name"
            ,"model_notes"
        )
        
        for name in string_properties:
            value = getattr(self, name)
            
            if value is not None:
                if not isinstance(value, str):
                    errors.append(f"{name} must be a single string.")
                elif value == "":
                    errors.append(f"{name} cannot be an empty string. Use None to leave it unset.")
         
        # Character vectors (x variables)
        for name in ("raw_variables", "x_variables"):
            value = getattr(self, name)
            
            if isinstance(value, str):
                values = (value,)
            else:
                try:
                    values = tuple(value)
                except TypeError:
                    errors.append(f"{name} must be a character vector.")
                    continue
                
            if not all(
                    item is None or isinstance(item, str)
                    for item in values
            ):
                    errors.append(f"{name} must be a character vector.")
        
        # Further checks on x variables
        x = [value for value in self.raw_variables if value is not None]
        
        if any(value =="" for value in x):
            errors.append("raw_variables must not contain empty strings.")
            
        duplicates = []
        for value in x:
            if x.count(value) > 1 and value not in duplicates:
                duplicates.append(value)
        
        if duplicates:
            errors.append("raw_variables contains duplicate name(s): " + ", ".join(duplicates) + ".")
        
        # Positive finite numerics
        for name in ("expo_val", "expo_0_rep"):
            value = getattr(self, name)
            
            if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value)
                or value <= 0
            ):
                errors.append(f"{name} must be a single positive finite number." f"(got {value}).")
        
        # Offset value    
        value = self.offset_value
        
        if value is not None:
            if (isinstance(value, (bool, np.bool)) or not isinstance(value, (int, float, np.number))):
                errors.append("offset_value must be a single number or None.")
            elif not np.isfinite(value):
                errors.append(f"offset_value must be finite(got {value}).")
        
        # Target & Predictions
        if (self.target_name is not None and self.preds_name is not None and self.target_name == self.preds_name):
            errors.append(f"target_name and preds_name are both '{self.target_name}'. " "They must refer to different columns.")
            
        # Columns required in train    
        if self.train is not None:
            columns = self.train.columns
            
            if (self.target_name is not None and self.target_name not in columns):
                errors.append(f"target_name '{self.target_name}' is not a column in the train set.")
                
            missing_x = [value for value in x if value not in columns]
            if missing_x: errors.append("raw_variables column(s) not found in train: " + ", ".join(missing_x) + ".")
            
            if (self.offset_name is not None and self.offset_name not in columns):
                errors.append(f"offset_name '{self.offset_name}' is not a column in train.")
                
        if errors:
            raise ValueError("\n".join(errors))
                