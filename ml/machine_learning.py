import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import  KNeighborsRegressor
from typing import List, Dict


def list_ml_models() -> Dict[str,List[str]]:
    """
    Lists all available classification and regression models from scikit-learn.
    
    Returns:
    A dictionary with two keys: 'classification' and 'regression', each containing a list of model names.
    """
    
    classification_models = ['random_forest', 'support_vector_machine', 'xgboost', 'tempcnn', 'mlp', 'lstm', 'resnet']
    regression_models = ['random_forest', 'support_vector_machine', 'knn']
    
    return {
        'classification': classification_models,
        'regression': regression_models
    }


def model_param_blueprint(model_name, task_type="classification") -> Dict:
    """
    Provides default parameters for a given model and task type, including an option to set the seed.
    
    Parameters:
    - model_name: String, the name of the model method.
    - task_type: String, either 'classification' or 'regression'.
    
    Returns:
    A dictionary with default parameters for the model.
    """
    default_params = {
        'classification': {
            'random_forest': {'n_estimators': 100, 'random_state': None},
            'support_vector_machine': {'C': 1.0, 'random_state': None}
        },
        'regression': {
            'random_forest': {'n_estimators': 100, 'random_state': None},
            'support_vector_machine': {'C': 1.0, 'random_state': None},
            'knn': {'n_neighbors': 5}
        }
    }
    
    # Check if task type is valid
    if task_type not in default_params:
        raise ValueError(f"Task type '{task_type}' is not valid. Choose 'classification' or 'regression'.")
    
    # Get default parameters for the specified model and task type
    params = default_params[task_type].get(model_name)
    
    if params is None:
        raise ValueError(f"Model '{model_name}' is not recognized for task type '{task_type}'.")
    
    return params


def ml_fit(predictor, label, ml_method="random_forest", parameters=None):
    """
    Fits a machine learning model to the provided predictor and label data.
    Automatically determines whether the task is classification or regression based on the label data.
    
    Parameters:
    - predictor: Features (predictor variables) as a DataFrame or NumPy array.
    - label: Target variable as a DataFrame, Series, or NumPy array.
    - ml_method: String specifying which machine learning method to use.
    - parameters: Dictionary of parameters to pass to the machine learning model constructor.
    
    Returns:
    The fitted machine learning model.
    """
    
    # Default parameters if none provided
    if parameters is None:
        parameters = {}
    
    # Determine if the task is classification or regression
    if np.issubdtype(label.dtype, np.number) and not pd.api.types.is_categorical_dtype(label):
        # Check if label is integer and has less than a certain number of unique values, could be classification
        if label.dtype == int and len(np.unique(label)) < np.sqrt(len(label)):
            task = "classification"
        else:
            task = "regression"
    else:
        task = "classification"
    
    # Dictionaries of supported machine learning methods for classification and regression
    classification_methods = {
        "random_forest": RandomForestClassifier,
        "support_vector_machine": SVC
    }
    
    regression_methods = {
        "random_forest": RandomForestRegressor,
        "support_vector_machine": SVR,
        "knn": KNeighborsRegressor
    }
    
    # Select the appropriate method dictionary based on the task
    methods = classification_methods if task == "classification" else regression_methods
    
    # Get the machine learning model class based on the method
    ml_model_class = methods.get(ml_method.lower())
    
    if not ml_model_class:
        raise ValueError(f"ML method '{ml_method}' is not supported for {task}.")
    
    # Create an instance of the model with the provided parameters
    model = ml_model_class(**parameters)
    
    # Fit the model
    model.fit(predictor, label)
    
    return model


def ml_predict(model, data):
    """
    Uses a trained model to make predictions on new data(datacube).

    Parameters:
    - model: The trained machine learning model.
    - data: The data cube containing the input features.

    Returns:
    The predictions made by the model.
    """
    try:
        # Making predictions
        predictions = model.predict(data)
        return predictions
    except Exception as e:
        # If something goes wrong, print an error message
        return f"An error occurred while making predictions: {e}"


def model_param_blueprint2(model_name, task_type="classification")-> Dict:
    """
    Provides default parameters and descriptions for a given model and task type, including an option to set the seed.
    
    Parameters:
    - model_name: String, the name of the model method.
    - task_type: String, either 'classification' or 'regression'.
    
    Returns:
    A dictionary with default parameters and their descriptions for the model.
    """
    model_params_descriptions = {
        'classification': {
            'random_forest': {
                'n_estimators': {
                    'default': 100,
                    'description': 'The number of trees in the forest.'
                },
                'seed': {
                    'default': None,
                    'description': 'Controls both the randomness of the bootstrapping of the samples used when building trees (if `bootstrap=True`) and the sampling of the features to consider when looking for the best split at each node.'
                }
            },
            # ....
        },
        'regression': {
            'random_forest': {
                'n_estimators': {
                    'default': 100,
                    'description': 'The number of trees in the forest.'
                },
                'seed': {
                    'default': None,
                    'description': 'Controls both the randomness of the bootstrapping of the samples used when building trees (if `bootstrap=True`) and the sampling of the features to consider when looking for the best split at each node.'
                }
            },
            # ....
        }
    }
    
    # Check if task type is valid
    if task_type not in model_params_descriptions:
        raise ValueError(f"Task type '{task_type}' is not valid. Choose 'classification' or 'regression'.")

    # Get default parameters and descriptions for the specified model and task type
    model_info = model_params_descriptions[task_type].get(model_name)
    
    if model_info is None:
        raise ValueError(f"Model '{model_name}' is not recognized for task type '{task_type}'.")

    return model_info


# cross-validation, fitting more than one model and returning the best model
# function for model parameter tuning purposes for ml_fit e.g. more than one learning rate