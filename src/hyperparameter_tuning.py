"""
Hyperparameter Tuning Module

Uses Optuna for Bayesian optimization of pooled LightGBM parameters.
Implements time series cross-validation for robust parameter selection.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesCrossValidator:
    """Time series cross-validation with expanding window"""

    def __init__(self,
                 n_splits: int = 3,
                 min_train_size: int = 500,
                 val_size: int = 30):
        """
        Initialize time series CV

        Args:
            n_splits: Number of CV splits
            min_train_size: Minimum training size
            val_size: Validation size for each split
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.val_size = val_size

    def split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/val splits with expanding window

        Args:
            df: DataFrame with date column

        Returns:
            List of (train_mask, val_mask) tuples
        """
        # Get unique dates sorted
        unique_dates = sorted(df['date'].unique())
        n_dates = len(unique_dates)

        # Calculate split points
        splits = []

        for i in range(self.n_splits):
            # Expanding window: increase training data for each split
            val_start_idx = self.min_train_size + (i * self.val_size)
            val_end_idx = val_start_idx + self.val_size

            if val_end_idx > n_dates:
                break

            # Get date ranges
            train_dates = unique_dates[:val_start_idx]
            val_dates = unique_dates[val_start_idx:val_end_idx]

            # Create masks
            train_mask = df['date'].isin(train_dates).values
            val_mask = df['date'].isin(val_dates).values

            splits.append((train_mask, val_mask))

        return splits


class LGBMHyperparameterTuner:
    """Hyperparameter tuning for pooled LightGBM using Optuna"""

    def __init__(self,
                 n_trials: int = 50,
                 cv_splits: int = 3,
                 min_train_size: int = 500,
                 val_size: int = 30,
                 seed: int = 42):
        """
        Initialize tuner

        Args:
            n_trials: Number of Optuna trials
            cv_splits: Number of CV splits
            min_train_size: Minimum training size for CV
            val_size: Validation size for CV
            seed: Random seed
        """
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.min_train_size = min_train_size
        self.val_size = val_size
        self.seed = seed
        self.best_params = {}
        self.study = None

        # Time series CV
        self.cv = TimeSeriesCrossValidator(
            n_splits=cv_splits,
            min_train_size=min_train_size,
            val_size=val_size
        )

    def tune(self,
            df: pd.DataFrame,
            feature_cols: List[str],
            verbose: bool = True) -> Dict:
        """
        Tune hyperparameters for pooled model (all categories together)

        Args:
            df: DataFrame with features (all categories)
            feature_cols: Feature columns (should include 'category')
            verbose: Print progress

        Returns:
            Best parameters
        """
        if verbose:
            print("\n" + "="*60)
            print("HYPERPARAMETER TUNING (POOLED MODEL)")
            print("="*60)
            print(f"  Trials: {self.n_trials}")
            print(f"  CV splits: {self.cv_splits}")
            print(f"  Features: {len(feature_cols)}")
            print(f"  Categories: {df['category'].nunique()}")

        # Ensure category is in features
        if 'category' not in feature_cols:
            feature_cols = feature_cols + ['category']

        # Create study
        sampler = TPESampler(seed=self.seed)
        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            study_name='lgbm_pooled'
        )

        # Define objective for pooled model
        def pooled_objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'seed': self.seed,
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 15.0),
            }

            cv_scores = []
            splits = self.cv.split(df)
            num_boost_round = trial.suggest_int('num_boost_round', 50, 500)

            for train_mask, val_mask in splits:
                X_train = df.loc[train_mask, feature_cols]
                y_train = df.loc[train_mask, 'ets_residual']
                X_val = df.loc[val_mask, feature_cols]
                y_val = df.loc[val_mask, 'ets_residual']

                # Handle missing values BEFORE categorical encoding
                X_train = X_train.ffill().bfill().fillna(0)
                X_val = X_val.ffill().bfill().fillna(0)

                # Handle categorical encoding AFTER fillna
                if 'category' in feature_cols:
                    X_train = X_train.copy()
                    X_val = X_val.copy()
                    X_train['category'] = X_train['category'].astype('category')
                    X_val['category'] = X_val['category'].astype('category')

                # Identify categorical features
                categorical_features = ['category'] if 'category' in feature_cols else []

                train_data = lgb.Dataset(
                    X_train, label=y_train,
                    categorical_feature=categorical_features
                )
                val_data = lgb.Dataset(
                    X_val, label=y_val,
                    reference=train_data,
                    categorical_feature=categorical_features
                )

                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[val_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=30, verbose=False),
                        lgb.log_evaluation(period=0)
                    ]
                )

                y_pred = model.predict(X_val, num_iteration=model.best_iteration)
                mae = np.mean(np.abs(y_val - y_pred))
                cv_scores.append(mae)

            return np.mean(cv_scores)

        # Optimize
        self.study.optimize(
            pooled_objective,
            n_trials=self.n_trials,
            show_progress_bar=verbose
        )

        # Get best parameters
        best_params = self.study.best_params
        self.best_params = best_params

        if verbose:
            print(f"\n  Best trial:")
            print(f"    MAE: {self.study.best_value:.2f}")
            print(f"  Best parameters:")
            for key, value in best_params.items():
                print(f"    {key}: {value}")

        return best_params


def save_tuned_params(params_dict: Dict, filepath: str = 'models/tuned_params.json'):
    """Save tuned parameters to JSON file"""
    import json
    import os

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(params_dict, f, indent=2)

    print(f"Tuned parameters saved to: {filepath}")


def load_tuned_params(filepath: str = 'models/tuned_params.json') -> Dict:
    """Load tuned parameters from JSON file"""
    import json

    with open(filepath, 'r') as f:
        params_dict = json.load(f)

    return params_dict
