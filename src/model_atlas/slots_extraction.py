from dataclasses import replace
from numbers import Number

import numpy as np
import pandas as pd

from .model_capsule import ModelCapsule

# Helper functions that check data types - checks needed before extracting slots from ModelCapsule.
# Model Capsule is immutable. Existing objects therefore are never modified in place.

def _check_model_capsule(x):
    """Check that x is a ModelCapsule item."""
    if not isinstance(x, ModelCapsule):
        raise TypeError("x must be a ModelCapsule object.")
        
def _check_fun(value, arg):
    """Check that value is a function"""
    if not callable(value):
        raise TypeError(f"{arg} must be a function.")
        
def _check_chr(value, arg):
    """Check that value represents a character vector."""
    if value is None:
        return
    
    if isinstance(value, str):
        return 
    
    if not isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        raise TypeError(f"{arg} must be a character vector.")
    
    if not all(item is None or isinstance(item, str) for item in value):
        raise TypeError(f"{arg} must be a character vector.")
        
def _check_scalar_chr(value, arg):
    """Check that value is a single string or missing character value."""
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{arg} must be a single string.")
        
def _check_tabular(value, arg):
    """Check that value is a pandas DataFrame."""
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{arg} must be a pandas DataFrame.")
        
def _check_scalar_numeric(value, arg):
    """Check that value represents a numeric."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{arg} must be a single number.")
        
    if not isinstance(value, Number):
        raise TypeError(f"{arg} must be a single number.")  
        
def _set_property(x, attribute, value):
    """Return a validated copy of the existing ModelCapsule with one attribute replaced."""
    _check_model_capsule(x)
    
    return replace(x, **{attribute: value})        
        
# Extracting the model object
def extract_model(x):
    """Extract the raw fitted model from a ModelCapsule."""
    _check_model_capsule(x)
    return x.model        

# Extracting the data sets used in the model.

def extract_train(x):
    """Extract the training set."""
    _check_model_capsule(x)
    return x.train

def extract_test(x):
    """Extract the test set."""
    _check_model_capsule(x)
    return x.test

def extract_validation(x):
    """Extract the validation set."""
    _check_model_capsule(x)
    return x.validation        
        
# Extract functions from the model pipeline

def extract_feat_eng_fun(x):
    """Extract the feature engineering function."""
    _check_model_capsule(x)
    return x.feat_eng_fun

def extract_pre_process_fun(x):
    """Extract the pre process function."""
    _check_model_capsule(x)
    return x.pre_process_fun

def extract_post_process_fun(x):
    """Extract the post process function."""
    _check_model_capsule(x)
    return x.post_process_fun
    
# Extract the raw and engineered variable names
def extract_raw_vars(x):
    """Extract the original-raw variables names before engineering."""
    _check_model_capsule(x)
    return [name for name in x.raw_variables if name is not None]   

def extract_engineered_vars(x):
    """Extract the final variables names fitted in the model after engineering."""
    _check_model_capsule(x)
    return [name for name in x.x_variables if name is not None] 

# Extract target variable and further useful slots of the model capsule. 
def extract_target(x):
    """Extract the name of the target variable."""
    _check_model_capsule(x)
    return x.target_name

def extract_preds_name(x):
    """Extract the name of the predictions as saved in the model."""
    _check_model_capsule(x)
    return x.preds_name

def extract_exposure_name(x):
    """Extract the name of the exposure variable that was used in the model."""
    _check_model_capsule(x)
    return x.exposure_name

def extract_model_name(x):
    """Extract the name assigned to the model."""
    _check_model_capsule(x)
    return x.model_name

def extract_model_notes(x):
    """Extract useful notes that were added to the ModelCapsule by the modeller."""
    _check_model_capsule(x)
    return x.model_notes


# Define setters. Each setter returns a new ModelCapsule. 

def set_model(x, value):
    """Return a copy with the fitted model replaced."""
    return _set_property(x, "model", value)

def set_train(x, value):
    """Return a new ModelCapsule with the training data replaced."""
    _check_tabular(value, "value")
    return _set_property(x, "train", value)
    
def set_test(x, value):
    """Return a new ModelCapsule with the test data replaced."""
    _check_tabular(value, "value")
    return _set_property(x, "test", value)
    
def set_validation(x, value):
    """Return a new ModelCapsule with the validation data replaced."""
    _check_tabular(value, "value")
    return _set_property(x, "validation", value)
        
def set_pre_process_fun(x, value):
    """Return a new ModelCapsule with the pre processing function replaced."""
    _check_fun(value, "value")
    return _set_property(x, "pre_process_fun", value)

def set_post_process_fun(x, value):
    """Return a new ModelCapsule with the post processing function replaced."""
    _check_fun(value, "value")
    return _set_property(x, "post_process_fun", value)

def set_feat_eng_fun(x, value):
    """Return a new ModelCapsule with the feature engineering function replaced."""
    _check_fun(value, "value")
    return _set_property(x, "feat_eng_fun", value)

def set_raw_variables(x, value):
    """Return a new ModelCapsule with the raw variables replaced."""
    _check_chr(value, "value")
    return _set_property(x, "raw_variables", value)

def set_x_variables(x, value):
    """Return a new ModelCapsule with the x (engineered) variables replaced."""
    _check_chr(value, "value")
    return _set_property(x, "x_variables", value)

def set_model_name(x, value):
    """Return a new ModelCapsule with the model name replaced."""
    _check_scalar_chr(value, "value")
    return _set_property(x, "model_name", value)

def set_model_notes(x, value):
    """Return a new ModelCapsule with the model notes replaced."""
    _check_scalar_chr(value, "value")
    return _set_property(x, "model_notes", value)
        
        
