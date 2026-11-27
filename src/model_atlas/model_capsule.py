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

def _model_predict(model, newdata):
    """Predict using any supported model and return a 1D float array. 
    *H2O models: predictions on h2o frames. For binary classification the positive-class probability is returned.
    *Binary classifiers with 'predict_proba': the positive-class probability.
    *Everything else: 'predict' statsmodels GLMs already predict on the response scale. 
    
    If 'predict()' returns a 2D result, the first column is used.
    
    Parameters
    ----------
    model: Any
        A fitted model. 
    newdata: pandas.DataFrame
        Data o predict on - already pre-processed and feature engineered.
        
    Returns
    -------
    numpy.ndarray
        One float prediction per row.
        
    Raises
    ------
    ValueError
        If the model is a multiclass classifier which has no single numeric prediction per row.
    ModuleNotFoundError
        If the model is an H2O model and the 'h2o' package is not installed.
    RuntimeError
        If the model's own prediction call fails - the original error is included in the message. 
    """
    nd = newdata if isinstance(newdata, pd.DataFrame) else pd.DataFrame(newdata)
    model_class = type(model).__name__
    
    multiclass_message = ("Multiclass models are not supported. Model diagnostics need a single numeric prediction per row."
                          "A multiclass model returns one probability per class.")
    
    if _is_h2o_model(model):
        import h2o
        
        hf = h2o.H2OFRAME(nd) #H2O required its own frame type and returns an H2O frame.
        try: 
            raw = model.predict(hf).as_data_frame()
        finally:
            h2o.remove(hf)
            
        if raw.shape[1] > 3:
            raise ValueError(multiclass_message)
        return raw.iloc[:, -1].to_numpy(dtype=float)
    
    feature_names = getattr(model, "feature_name_in_", None)
    if feature_names is not None and all(name in nd.columns for name in feature_names):
        nd = nd[list(feature_names)]
        
    classes = getattr(model, "classes_", None)
    is_classifier = hasattr(model, "predict_proba") and classes is not None

    if is_classifier and len(classes) > 2:
        raise ValueError(multiclass_message)
        
    try:
        if is_classifier:
            raw = model.predict_proba(nd)[:, -1]
        else:
            raw = model.predict(nd)
    except Exception as error:
        raise RuntimeError(f"predict() failed for model class '{model_class}'.\n{error}") from error
        
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
        
    raw = np.asarray(raw)
    if raw.ndim == 2:
        raw = raw[:, 0]
        
    return raw.astype(float).ravel()

def _format_nrow(df):
    """Row count with thousands separators or <not set> for None."""
    return "<not set>" if df is None else f"{len(df):,}"

def _format_number(value):
    """Format a number"""
    if value is None:
        return "NA"
    return f"{value:.15g}"
    
@dataclass(frozen = True)
class ModelCapsule:
    """Model-agnostic container for a fitted model and its pipeline.
    
    Only 'model' is required, every other attribute has a default. All attributes are validated when the capsule
    is created. 
    
    Parameters
    ----------
    model: Any
        A fitted model object with a 'predict' method.
    train, test, validation: pandas.DataFrame or None, default None
        Datasets.
    feat_eng_fun: callable, default identity
        Feature engineering pipeline.
    pre_process_fun: callable, default identity
        Pre-processing pipeline
    post_process_fun: callable, default identity
        Post-processing pipeline
    raw_variables: str or sequenec of str, default ()
        Original input feature names. A single string is accepted. An empty tuple means 'not set'.
    x_variables: str or sequence of str, default ()
        Engineered feature names. A single string is accepted. An empty tuple means 'not set'.
    target_name: str or None, default None
        Target variable name.
    preds_name: str or None, default None
        Prediction column name. 
    exposure_name: str or None, default 'exposure'
        Exposure column name. It does not have to exist in 'train'. Unit weights are used when it is absent. 
    exposure_val: float, default 1
        Reference exposure value.
    expo_0_rep: float, default 0.1
        Replacement for zero-exposure rows.
    offset_name: str or None, default None
        Offset variable name. 
    offset_value: float or None, default None
        Offset value.
    model_name: str or None, default None
        Label for the model.
    model_notes: str or None, default None
        Notes about the model - for example deployment notes. 
        
    Raises
    ------
    ValueError
        If any attribute is invalid. All problems are listed together one per line. See '_validate' for the checks.
        
    Example
    -------
    from sklearn.linear_model import LinearRegression
    df = pd.DataFrame({"x": [1,2,3,4], "y":[2.0, 4.1, 5.9, 8.2]})
    mc = ModelCapsule(
        model = LinearRegression().fit(df[["x"]], df[["y"]]),
        train = df,
        target_name = "y",
        raw_variables = "x",
        model_name = "Model_LR"
        ...
        )
    mc.raw_variables
    ('x',)
    """
    
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
    exposure_val: float = 1
    expo_0_rep: float = 0.1
    
    offset_name: str | None = None
    offset_value: float | None = None
    
    model_name: str | None = None
    model_notes: str | None = None
    
    def __post_init__(self):
        for name in ("raw_variables", "x_variables"):
            object.__setattr__(self, name, _as_name_tuple(getattr(self, name)))
        self._validate()
    
    # Components check/validation    
    def _validate(self):
        """Check every attribute and raise one error listing all problems. 
        
        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If any of these checks fail:
            
            * 'model' is None.
            * 'train', 'test', 'validation' is not a DataFrame or None.
            * A pipeline function is not callable.
            * A name attribute is not a single string or None or is "".
            * 'raw_variables' or "x_variables" is not a character vector.
            * 'raw_variables' contains empty strings or duplicates.
            * 'expo_val' or 'expo_0_rep' is not a poistive finite number.
            * 'offset_value' is not a finite number or None. 
            * 'target_name' and 'preds_name' are the same. 
            * 'target_name', 'raw_variables' or 'offset_name' are not columns of 'train'
        """
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
        valid_vectors = {}
        for name in ("raw_variables", "x_variables"):
            value = getattr(self, name)
            is_vector = isinstance(value, tuple) and all(item is None or isinstance(item, str) for item in value)
            
            if not is_vector:
                errors.append(f"{name} must be a string or a sequence of strings.")
            valid_vectors[name] = is_vector
            
        x = []
        if valid_vectors["raw_variables"]:
            x = [value for value in self.raw_variables if value is not None]
        
        if any(value =="" for value in x):
            errors.append("raw variables must not contain empty strings.")

        seen = set()
        duplicates = []
        for value in x:
            if value in seen and value not in duplicates:
                duplicates.append(value)
            seen.add(value)
        
        if duplicates:
            errors.append("raw_variables contains duplicate name(s): " + ", ".join(duplicates) + ".")
        
        # Positive finite numerics
        for name in ("exposure_val", "expo_0_rep"):
            value = getattr(self, name)
            
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or value <= 0):
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
                