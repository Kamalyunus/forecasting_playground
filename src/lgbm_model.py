"""
LightGBM Model Module

Trains a single pooled LightGBM model on ETS residuals from all categories.
Since ETS captures category-specific trend/seasonality, the residuals represent
common patterns (promotions, weather, holidays) that benefit from pooled learning.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LGBMResidualModel:
    """Pooled LightGBM model for forecasting ETS residuals across all categories"""

    def __init__(self, params: Optional[Dict] = None, tuned_params: Optional[Dict] = None):
        """
        Initialize pooled LightGBM model

        Args:
            params: LightGBM parameters (default: conservative params)
            tuned_params: Optuna-tuned parameters (dict with 'pooled' key)
        """
        if params is None:
            self.default_params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'max_depth': 5,
                'learning_rate': 0.05,
                'min_data_in_leaf': 20,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'verbose': -1,
                'seed': 42
            }
        else:
            self.default_params = params

        self.tuned_params = tuned_params
        self.model: Optional[lgb.Booster] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self.train_history: Optional[Dict] = None

    def prepare_features(self,
                        df: pd.DataFrame,
                        exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare feature matrix for training (includes category as feature)

        Args:
            df: DataFrame with all features
            exclude_cols: Additional columns to exclude

        Returns:
            X (features), feature_names
        """
        # Default exclusions
        default_exclude = [
            'date', 'sales',
            'ets_fitted', 'ets_residual',
            'actual',
            'holiday_name',
            'num_skus',  # Exclude raw SKU count (use pct_skus_on_high_impact_promo instead)
            'high_impact_promo',  # Exclude raw promo count (use pct_skus_on_high_impact_promo instead)
            'instock_rate'  # Exclude instock rate (not needed as feature)
        ]

        if exclude_cols:
            default_exclude.extend(exclude_cols)

        # Get feature columns (including category)
        feature_cols = [col for col in df.columns if col not in default_exclude]

        # Handle missing values (forward fill, then backward fill, then 0)
        X = df[feature_cols].copy()
        X = X.ffill().bfill().fillna(0)

        # Encode category as categorical
        if 'category' in X.columns:
            X['category'] = X['category'].astype('category')

        return X, feature_cols

    def train(self,
             df: pd.DataFrame,
             train_mask: np.ndarray,
             val_mask: np.ndarray,
             num_boost_round: int = 300,
             early_stopping_rounds: int = 30) -> lgb.Booster:
        """
        Train pooled LightGBM model on all categories

        Args:
            df: DataFrame with features and ETS residuals (all categories)
            train_mask: Boolean mask for training data
            val_mask: Boolean mask for validation data
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping patience

        Returns:
            Trained LightGBM model
        """
        print("="*60)
        print("TRAINING POOLED LIGHTGBM MODEL")
        print("="*60)
        print(f"\n  Training on all categories combined...")

        # Prepare features (include category as a feature)
        X, feature_names = self.prepare_features(df)
        y = df['ets_residual'].values

        # Split train/val
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]

        print(f"    Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        print(f"    Features: {len(feature_names)} (including category)")
        print(f"    Categories: {df['category'].nunique()}")
        print(f"    Target (residual) - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")

        # Identify categorical features
        categorical_features = ['category'] if 'category' in feature_names else []

        # Create datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=feature_names,
            categorical_feature=categorical_features
        )
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            reference=train_data,
            feature_name=feature_names,
            categorical_feature=categorical_features
        )

        # Get parameters (check for tuned parameters)
        if self.tuned_params and 'pooled' in self.tuned_params:
            params = self.default_params.copy()
            tuned = self.tuned_params['pooled'].copy()
            if 'num_boost_round' in tuned:
                num_boost_round = tuned.pop('num_boost_round')
            params.update(tuned)
            print(f"    Using tuned hyperparameters")
        else:
            params = self.default_params
            print(f"    Using default hyperparameters")

        # Train model
        evals_result = {}
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=50),
                lgb.record_evaluation(evals_result)
            ]
        )

        # Store model and results
        self.model = model
        self.train_history = evals_result

        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        self.feature_importance = importance_df

        # Print results
        best_iteration = model.best_iteration
        train_mae = evals_result['train']['l1'][best_iteration - 1]
        val_mae = evals_result['val']['l1'][best_iteration - 1]

        print(f"    Best iteration: {best_iteration}")
        print(f"    Train MAE: {train_mae:.2f}")
        print(f"    Val MAE: {val_mae:.2f}")
        print(f"    Top 5 features: {', '.join(importance_df.head(5)['feature'].tolist())}")
        print(f"    ✓ Pooled LightGBM trained successfully")

        print("\n" + "="*60)
        print("LIGHTGBM TRAINING COMPLETE")
        print("="*60)

        return model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict residuals using pooled model

        Args:
            df: DataFrame with features (all categories)

        Returns:
            Predicted residuals
        """
        if self.model is None:
            raise ValueError("No trained model found. Call train() first.")

        # Prepare features (include category)
        X, _ = self.prepare_features(df)

        # Predict
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)

        return predictions


def split_train_val_test(df: pd.DataFrame,
                         train_days: int = 640,
                         val_days: int = 30,
                         test_days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test by date

    Args:
        df: DataFrame with 'date' column
        train_days: Number of days for training
        val_days: Number of days for validation
        test_days: Number of days for testing

    Returns:
        train_df, val_df, test_df
    """
    # Get unique dates sorted
    unique_dates = sorted(df['date'].unique())

    # Calculate split points
    train_end_idx = train_days
    val_end_idx = train_end_idx + val_days
    test_end_idx = val_end_idx + test_days

    if test_end_idx > len(unique_dates):
        raise ValueError(
            f"Not enough data: need {test_end_idx} days, have {len(unique_dates)}"
        )

    # Get date ranges
    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:test_end_idx]

    # Create splits
    train_df = df[df['date'].isin(train_dates)].copy()
    val_df = df[df['date'].isin(val_dates)].copy()
    test_df = df[df['date'].isin(test_dates)].copy()

    return train_df, val_df, test_df
