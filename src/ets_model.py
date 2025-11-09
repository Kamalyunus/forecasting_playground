"""
ETS (Exponential Smoothing) Model Module

Fits ETS models to capture structured seasonality and trend.
Extracts components for use in hybrid forecasting approach.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ETSDecomposer:
    """ETS model fitting and decomposition for category-level forecasting"""

    def __init__(self,
                 seasonal_periods: int = 365,
                 trend: str = 'add',
                 seasonal: str = 'mul',
                 damped_trend: bool = True,
                 smoothing_level: Optional[float] = None,
                 smoothing_trend: Optional[float] = None,
                 smoothing_seasonal: Optional[float] = None):
        """
        Initialize ETS decomposer

        Args:
            seasonal_periods: Number of periods in seasonal cycle (365 for yearly)
            trend: Trend type ('add', 'mul', or None)
            seasonal: Seasonal type ('add', 'mul', or None)
            damped_trend: Use damped trend
            smoothing_level: Alpha parameter (higher = more weight to recent, 0-1)
            smoothing_trend: Beta parameter (higher = more weight to recent, 0-1)
            smoothing_seasonal: Gamma parameter (higher = more weight to recent, 0-1)
        """
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.smoothing_seasonal = smoothing_seasonal

        # Storage for fitted models and results
        self.models: Dict[str, ExponentialSmoothing] = {}
        self.fitted_models: Dict[str, object] = {}
        self.results: Dict[str, pd.DataFrame] = {}

    def fit_category(self,
                    df: pd.DataFrame,
                    category: str,
                    target_col: str = 'sales') -> Tuple[object, pd.DataFrame]:
        """
        Fit ETS model for a single category

        Args:
            df: Category-day DataFrame
            category: Category name
            target_col: Target column (sales)

        Returns:
            fitted_model, results_df
        """
        # Filter data for category
        cat_df = df[df['category'] == category].copy()
        cat_df = cat_df.sort_values('date').reset_index(drop=True)

        # Get time series
        y = cat_df[target_col].values

        print(f"\n  Fitting ETS for {category}...")
        print(f"    Data points: {len(y)}")
        print(f"    Date range: {cat_df['date'].min().date()} to {cat_df['date'].max().date()}")
        print(f"    Mean: {y.mean():.2f}, Std: {y.std():.2f}")

        try:
            # Create and fit model
            model = ExponentialSmoothing(
                y,
                seasonal_periods=self.seasonal_periods,
                trend=self.trend,
                seasonal=self.seasonal,
                damped_trend=self.damped_trend,
                initialization_method='estimated'
            )

            # Fit with specified smoothing parameters or optimize
            if self.smoothing_level is not None and self.smoothing_trend is not None and self.smoothing_seasonal is not None:
                # Use specified smoothing parameters (all must be provided)
                fit_kwargs = {
                    'smoothing_level': self.smoothing_level,
                    'smoothing_trend': self.smoothing_trend,
                    'smoothing_seasonal': self.smoothing_seasonal,
                    'optimized': False
                }
                # If using damped trend, need to provide damping parameter
                if self.damped_trend:
                    fit_kwargs['damping_trend'] = 0.98  # Standard damping value

                fitted_model = model.fit(**fit_kwargs)
                print(f"    Using specified smoothing: α={self.smoothing_level}, β={self.smoothing_trend}, γ={self.smoothing_seasonal}")
            else:
                # Optimize smoothing parameters
                fitted_model = model.fit(optimized=True, use_brute=False)
                print(f"    Optimized smoothing: α={fitted_model.params['smoothing_level']:.3f}, "
                      f"β={fitted_model.params.get('smoothing_trend', 0):.3f}, "
                      f"γ={fitted_model.params.get('smoothing_seasonal', 0):.3f}")

            # Extract components
            fitted_values = fitted_model.fittedvalues
            residuals = y - fitted_values

            # Get level (trend) component
            # Note: level is the smoothed level component
            level = fitted_model.level

            # Get seasonal component
            seasonal = fitted_model.season

            # Create results DataFrame
            results_df = pd.DataFrame({
                'date': cat_df['date'],
                'category': category,
                'actual': y,
                'ets_fitted': fitted_values,
                'ets_residual': residuals,
                'ets_level': level,
            })

            # Store model and results
            self.models[category] = model
            self.fitted_models[category] = fitted_model
            self.results[category] = results_df

            # Calculate fit statistics
            mse = np.mean(residuals**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(residuals))
            mape = np.mean(np.abs(residuals / y)) * 100

            print(f"    Model fit - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
            print(f"    ✓ ETS model fitted successfully")

            return fitted_model, results_df

        except Exception as e:
            print(f"    ✗ Error fitting ETS model: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"    Using simple moving average fallback...")

            # Fallback: Use simple moving average
            fitted_values = pd.Series(y).rolling(self.seasonal_periods, min_periods=1).mean().values
            residuals = y - fitted_values
            level = fitted_values

            results_df = pd.DataFrame({
                'date': cat_df['date'],
                'category': category,
                'actual': y,
                'ets_fitted': fitted_values,
                'ets_residual': residuals,
                'ets_level': level,
            })

            # Store None model and results for fallback forecasting
            self.fitted_models[category] = None
            self.results[category] = results_df

            return None, results_df

    def fit_all_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit ETS models for all categories

        Args:
            df: Category-day DataFrame with features

        Returns:
            DataFrame with ETS components added
        """
        print("="*60)
        print("FITTING ETS MODELS")
        print("="*60)

        categories = df['category'].unique()
        print(f"Categories to fit: {list(categories)}")

        all_results = []

        for category in categories:
            fitted_model, results_df = self.fit_category(df, category)
            all_results.append(results_df)

        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)

        # Merge with original DataFrame
        df_with_ets = df.merge(
            combined_results[['date', 'category', 'ets_fitted', 'ets_residual', 'ets_level']],
            on=['date', 'category'],
            how='left'
        )

        print("\n" + "="*60)
        print("ETS FITTING COMPLETE")
        print("="*60)
        print(f"Categories fitted: {len(self.fitted_models)}")
        print(f"ETS components added: ets_fitted, ets_residual, ets_level")

        return df_with_ets

    def forecast(self,
                category: str,
                steps: int = 30) -> np.ndarray:
        """
        Generate ETS forecast for a category

        Args:
            category: Category name
            steps: Number of steps to forecast

        Returns:
            Forecast array
        """
        if category not in self.fitted_models:
            raise ValueError(f"No fitted model for category: {category}")

        fitted_model = self.fitted_models[category]

        if fitted_model is None:
            # Fallback: Use last known level
            last_level = self.results[category]['ets_level'].iloc[-1]
            forecast = np.full(steps, last_level)
        else:
            forecast = fitted_model.forecast(steps=steps)

        return forecast

    def forecast_all_categories(self, steps: int = 30) -> Dict[str, np.ndarray]:
        """
        Generate forecasts for all categories

        Args:
            steps: Number of steps to forecast

        Returns:
            Dictionary of forecasts by category
        """
        forecasts = {}

        for category in self.fitted_models.keys():
            forecasts[category] = self.forecast(category, steps)

        return forecasts

    def get_residuals(self, category: str) -> pd.Series:
        """
        Get residuals for a category

        Args:
            category: Category name

        Returns:
            Residuals series
        """
        if category not in self.results:
            raise ValueError(f"No results for category: {category}")

        return self.results[category]['ets_residual']


def fit_ets_models(df: pd.DataFrame,
                   seasonal_periods: int = 365,
                   trend: str = 'add',
                   seasonal: str = 'mul',
                   damped_trend: bool = True,
                   smoothing_level: Optional[float] = None,
                   smoothing_trend: Optional[float] = None,
                   smoothing_seasonal: Optional[float] = None) -> Tuple[pd.DataFrame, ETSDecomposer]:
    """
    Convenience function to fit ETS models

    Args:
        df: Category-day DataFrame with features
        seasonal_periods: Seasonal period (default 365 for yearly)
        trend: Trend type
        seasonal: Seasonal type
        damped_trend: Use damped trend
        smoothing_level: Alpha parameter (higher = more weight to recent, 0-1)
        smoothing_trend: Beta parameter (higher = more weight to recent, 0-1)
        smoothing_seasonal: Gamma parameter (higher = more weight to recent, 0-1)

    Returns:
        DataFrame with ETS components, ETSDecomposer object
    """
    decomposer = ETSDecomposer(
        seasonal_periods=seasonal_periods,
        trend=trend,
        seasonal=seasonal,
        damped_trend=damped_trend,
        smoothing_level=smoothing_level,
        smoothing_trend=smoothing_trend,
        smoothing_seasonal=smoothing_seasonal
    )

    df_with_ets = decomposer.fit_all_categories(df)

    return df_with_ets, decomposer
