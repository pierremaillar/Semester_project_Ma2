from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import time

def train_random_forest(x_train, y_train, x_test, y_test, parameters=None):
    """
    Trains a Random Forest Classifier and evaluates its performance.

    Args:
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        x_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing labels.
        parameters (dict, optional): Parameters for the RandomForestClassifier.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
    """
    tic = time.time()

    # Set default parameters if not provided
    if parameters is None:
        parameters = {'n_estimators': 50,
                      'max_depth': 600,
                      'max_features': 600,
                      'bootstrap': False,
                      'class_weight': 'balanced',
                      'min_samples_leaf': 5}

    # Create a Random Forest Classifier with the specified parameters
    model = RandomForestClassifier(**parameters)
    print(f"Parameters are: {parameters}")

    # Fit the model to the training data
    model.fit(x_train, y_train)
    print("Training done")
    return model
