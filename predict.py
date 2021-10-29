def predict_hotdog(trained_model, input_features):
    """
    Predict hotdog or no hotdog.

    Args:
        trained_model (scikit-learn estimator): trained classifier
        input_features (image) : something here.
    Returns:
        (str) the model's prediction
    """
    return trained_model.predict(input_features.reshape(1, -1))