from ml.machine_learning import ml_fit, ml_predict, list_ml_models, model_param_blueprint


models = list_ml_models()
print(models['classification'])


rf_classification_model_params = model_param_blueprint('random_forest', 'classification')
print("\nRandom forest Classification model default parameters:")
print(rf_classification_model_params)