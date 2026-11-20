from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

def _identity(df):
    return df

def _identity_post(preds, df_raw):
    return preds

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
                