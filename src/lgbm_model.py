"""
LightGBM Model Module

Trains LightGBM on ETS residuals to capture complex patterns:
- Promotions and events
- Weather impacts
- Holiday effects
- Other irregular patterns
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LGBMResidualModel:
    """LightGBM model for forecasting ETS residuals"""

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize LightGBM model

        Args:
            params: LightGBM parameters (default: MVP conservative params)
        """
        if params is None:
            self.params = {
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
            self.params = params

        self.models: Dict[str, lgb.Booster] = {}
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        self.train_history: Dict[str, Dict] = {}

    def prepare_features(self,
                        df: pd.DataFrame,
                        exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare feature matrix for training

        Args:
            df: DataFrame with all features
            exclude_cols: Additional columns to exclude

        Returns:
            X (features), feature_names
        """
        # Default exclusions
        default_exclude = [
            'date', 'category', 'sales', 'oos_adjusted_sales',
            'ets_fitted', 'ets_residual',  # Don't use fitted/residual as features
            'actual',  # From ETS results
            'holiday_name'  # Categorical, needs encoding if used
        ]

        if exclude_cols:
            default_exclude.extend(exclude_cols)

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in default_exclude]

        # Handle missing values (forward fill, then backward fill, then 0)
        X = df[feature_cols].copy()
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)

        return X, feature_cols

    def train_category(self,
                      df: pd.DataFrame,
                      category: str,
                      train_mask: np.ndarray,
                      val_mask: np.ndarray,
                      num_boost_round: int = 300,
                      early_stopping_rounds: int = 30) -> lgb.Booster:
        """
        Train LightGBM for a single category

        Args:
            df: DataFrame with features and ETS residuals
            category: Category name
            train_mask: Boolean mask for training data
            val_mask: Boolean mask for validation data
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping patience

        Returns:
            Trained LightGBM model
        """
        print(f"\n  Training LightGBM for {category}...")

        # Filter category data
        cat_df = df[df['category'] == category].copy()

        # Get masks for this category
        cat_train_mask = train_mask[df['category'] == category]
        cat_val_mask = val_mask[df['category'] == category]

        # Prepare features
        X, feature_names = self.prepare_features(cat_df)
        y = cat_df['ets_residual'].values

        # Split train/val
        X_train = X[cat_train_mask]
        y_train = y[cat_train_mask]
        X_val = X[cat_val_mask]
        y_val = y[cat_val_mask]

        print(f"    Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        print(f"    Features: {len(feature_names)}")
        print(f"    Target (residual) - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names)

        # Train model
        evals_result = {}
        model = lgb.train(
            self.params,
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
        self.models[category] = model
        self.train_history[category] = evals_result

        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        self.feature_importance[category] = importance_df

        # Print results
        best_iteration = model.best_iteration
        train_mae = evals_result['train']['l1'][best_iteration - 1]
        val_mae = evals_result['val']['l1'][best_iteration - 1]

        print(f"    Best iteration: {best_iteration}")
        print(f"    Train MAE: {train_mae:.2f}")
        print(f"    Val MAE: {val_mae:.2f}")
        print(f"    Top 5 features: {', '.join(importance_df.head(5)['feature'].tolist())}")
        print(f"    ✓ LightGBM trained successfully")

        return model

    def train_all_categories(self,
                            df: pd.DataFrame,
                            train_mask: np.ndarray,
                            val_mask: np.ndarray,
                            **kwargs) -> None:
        """
        Train LightGBM for all categories

        Args:
            df: DataFrame with features and ETS residuals
            train_mask: Boolean mask for training data
            val_mask: Boolean mask for validation data
            **kwargs: Additional arguments for train_category
        """
        print("="*60)
        print("TRAINING LIGHTGBM MODELS")
        print("="*60)

        categories = df['category'].unique()
        print(f"Categories to train: {list(categories)}")

        for category in categories:
            self.train_category(df, category, train_mask, val_mask, **kwargs)

        print("\n" + "="*60)
        print("LIGHTGBM TRAINING COMPLETE")
        print("="*60)
        print(f"Models trained: {len(self.models)}")

    def predict_category(self,
                        df: pd.DataFrame,
                        category: str) -> np.ndarray:
        """
        Predict residuals for a category

        Args:
            df: DataFrame with features
            category: Category name

        Returns:
            Predicted residuals
        """
        if category not in self.models:
            raise ValueError(f"No trained model for category: {category}")

        # Filter category data
        cat_df = df[df['category'] == category].copy()

        # Prepare features
        X, _ = self.prepare_features(cat_df)

        # Predict
        model = self.models[category]
        predictions = model.predict(X, num_iteration=model.best_iteration)

        return predictions

    def predict_all_categories(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict residuals for all categories

        Args:
            df: DataFrame with features

        Returns:
            Predicted residuals for entire dataframe
        """
        predictions = np.zeros(len(df))

        for category in df['category'].unique():
            mask = df['category'] == category
            predictions[mask] = self.predict_category(df, category)

        return predictions

    def plot_feature_importance(self,
                               category: str,
                               top_n: int = 20) -> None:
        """
        Plot feature importance for a category

        Args:
            category: Category name
            top_n: Number of top features to plot
        """
        import matplotlib.pyplot as plt

        if category not in self.feature_importance:
            raise ValueError(f"No feature importance for category: {category}")

        importance_df = self.feature_importance[category].head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importance_df['feature'], importance_df['importance'])
        ax.set_xlabel('Importance (Gain)')
        ax.set_title(f'{category}: Top {top_n} Most Important Features')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'outputs/plots/feature_importance_{category}.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved: outputs/plots/feature_importance_{category}.png")
        plt.close()

    def plot_training_history(self, category: str) -> None:
        """
        Plot training history (loss curves)

        Args:
            category: Category name
        """
        import matplotlib.pyplot as plt

        if category not in self.train_history:
            raise ValueError(f"No training history for category: {category}")

        history = self.train_history[category]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history['train']['l1'], label='Train MAE', alpha=0.7)
        ax.plot(history['val']['l1'], label='Val MAE', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('MAE')
        ax.set_title(f'{category}: Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'outputs/plots/training_history_{category}.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved: outputs/plots/training_history_{category}.png")
        plt.close()

    def get_feature_importance_summary(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance summary across all categories

        Args:
            top_n: Number of top features per category

        Returns:
            Summary DataFrame
        """
        summaries = []

        for category, importance_df in self.feature_importance.items():
            top_features = importance_df.head(top_n).copy()
            top_features['category'] = category
            top_features['rank'] = range(1, len(top_features) + 1)
            summaries.append(top_features)

        combined = pd.concat(summaries, ignore_index=True)

        return combined


def split_train_val_test(df: pd.DataFrame,
                         train_days: int = 640,
                         val_days: int = 30,
                         test_days: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/val/test splits (simple time-based split)

    Args:
        df: DataFrame with date column
        train_days: Number of training days
        val_days: Number of validation days
        test_days: Number of test days

    Returns:
        train_mask, val_mask, test_mask
    """
    # Ensure sorted by date
    df = df.sort_values(['category', 'date']).reset_index(drop=True)

    # Get unique dates
    unique_dates = sorted(df['date'].unique())
    total_days = len(unique_dates)

    print(f"\nCreating train/val/test splits...")
    print(f"  Total unique dates: {total_days}")
    print(f"  Train: {train_days} days")
    print(f"  Val: {val_days} days")
    print(f"  Test: {test_days} days")

    # Define split points
    train_end_idx = train_days
    val_end_idx = train_days + val_days

    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:val_end_idx + test_days]

    # Create masks
    train_mask = df['date'].isin(train_dates).values
    val_mask = df['date'].isin(val_dates).values
    test_mask = df['date'].isin(test_dates).values

    print(f"\nSplit summary:")
    print(f"  Train: {train_mask.sum()} records ({train_dates[0].date()} to {train_dates[-1].date()})")
    print(f"  Val: {val_mask.sum()} records ({val_dates[0].date()} to {val_dates[-1].date()})")
    print(f"  Test: {test_mask.sum()} records ({test_dates[0].date()} to {test_dates[-1].date()})")

    return train_mask, val_mask, test_mask


if __name__ == "__main__":
    # Test LightGBM module
    print("Testing LightGBM module...")

    # Load data with ETS components
    df = pd.read_csv("data/category_day_with_ets.csv")
    df['date'] = pd.to_datetime(df['date'])

    # Create splits
    train_mask, val_mask, test_mask = split_train_val_test(df)

    # Train LightGBM
    lgbm_model = LGBMResidualModel()
    lgbm_model.train_all_categories(df, train_mask, val_mask)

    # Generate predictions
    predictions = lgbm_model.predict_all_categories(df)
    df['lgbm_residual_pred'] = predictions

    # Save
    output_path = "data/category_day_with_lgbm.csv"
    df.to_csv(output_path, index=False)
    print(f"\nData with LGBM predictions saved to: {output_path}")

    # Plot feature importance
    print("\nGenerating feature importance plots...")
    for category in df['category'].unique():
        lgbm_model.plot_feature_importance(category)
        lgbm_model.plot_training_history(category)
