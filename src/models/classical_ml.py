"""
Classical ML models for environmental sound classification.

Models:
- SVM (RBF kernel) with GridSearchCV
- Random Forest with hyperparameter tuning

Both models are evaluated on Pipeline A (raw) and Pipeline B (DSP-processed) features.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import config


def create_svm_pipeline():
    """Create SVM pipeline with scaling and hyperparameter grid."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=config.SVM_KERNEL, random_state=config.RANDOM_SEED, probability=True))
    ])

    param_grid = {
        'svm__C': config.SVM_C_RANGE,
        'svm__gamma': config.SVM_GAMMA_RANGE,
    }

    return pipeline, param_grid


def create_random_forest():
    """Create Random Forest with hyperparameter grid."""
    model = RandomForestClassifier(random_state=config.RANDOM_SEED, n_jobs=-1)

    param_grid = {
        'n_estimators': config.RF_N_ESTIMATORS,
        'max_depth': config.RF_MAX_DEPTH,
    }

    return model, param_grid


def train_with_grid_search(model, param_grid, X_train, y_train, cv=3):
    """
    Train model with GridSearchCV.

    Args:
        model: sklearn estimator or pipeline.
        param_grid: Dict of hyperparameters to search.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of cross-validation folds for grid search.

    Returns:
        best_model: Fitted model with best parameters.
        best_params: Dict of best hyperparameters.
        best_score: Best cross-validation score.
    """
    grid = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy',
                        n_jobs=-1, verbose=0, refit=True)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def train_svm(X_train, y_train, cv=3):
    """Train SVM with grid search."""
    pipeline, param_grid = create_svm_pipeline()
    return train_with_grid_search(pipeline, param_grid, X_train, y_train, cv)


def train_random_forest(X_train, y_train, cv=3):
    """Train Random Forest with grid search."""
    model, param_grid = create_random_forest()

    # Scale features for consistency
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    best_model, best_params, best_score = train_with_grid_search(
        model, param_grid, X_scaled, y_train, cv)

    return best_model, best_params, best_score, scaler


def get_feature_importance(rf_model, feature_names=None):
    """Extract feature importance from trained Random Forest."""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    if feature_names is not None:
        return [(feature_names[i], importances[i]) for i in indices]
    return [(i, importances[i]) for i in indices]
