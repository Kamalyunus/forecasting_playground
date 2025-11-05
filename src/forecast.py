"""
Forecast Generation Module

Combines ETS and LightGBM forecasts for final prediction.
Handles feature generation for future periods.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import timedelta


class HybridForecaster:
    """Hybrid forecasting combining ETS and LightGBM"""

    def __init__(self, ets_decomposer, lgbm_model):
        """
        Initialize hybrid forecaster

        Args:
            ets_decomposer: Fitted ETSDecomposer
            lgbm_model: Trained LGBMResidualModel
        """
        self.ets_decomposer = ets_decomposer
        self.lgbm_model = lgbm_model

    def create_future_features(self,
                              historical_df: pd.DataFrame,
                              category: str,
                              forecast_horizon: int = 30,
                              future_promo_plan: Optional[pd.DataFrame] = None,
                              future_weather: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create feature matrix for future forecast period

        Args:
            historical_df: Historical data with all features
            category: Category name
            forecast_horizon: Number of days to forecast
            future_promo_plan: Future promotion plan (optional)
            future_weather: Future weather forecast (optional)

        Returns:
            DataFrame with future features
        """
        # Get category historical data
        cat_hist = historical_df[historical_df['category'] == category].copy()
        cat_hist = cat_hist.sort_values('date').reset_index(drop=True)

        # Last date in history
        last_date = cat_hist['date'].max()

        # Create future dates
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_horizon,
            freq='D'
        )

        # Initialize future dataframe
        future_df = pd.DataFrame({
            'date': future_dates,
            'category': category
        })

        # ========================================================================
        # TEMPORAL FEATURES (can be calculated directly)
        # ========================================================================
        future_df['day_of_week'] = future_df['date'].dt.dayofweek
        future_df['week_of_year'] = future_df['date'].dt.isocalendar().week.astype(int)
        future_df['month'] = future_df['date'].dt.month
        future_df['day_of_year'] = future_df['date'].dt.dayofyear
        future_df['week_of_month'] = ((future_df['date'].dt.day - 1) // 7 + 1)
        future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)

        # Cyclical encoding
        future_df['day_of_week_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
        future_df['day_of_week_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)
        future_df['week_of_year_sin'] = np.sin(2 * np.pi * future_df['week_of_year'] / 52)
        future_df['week_of_year_cos'] = np.cos(2 * np.pi * future_df['week_of_year'] / 52)
        future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
        future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
        future_df['day_of_year_sin'] = np.sin(2 * np.pi * future_df['day_of_year'] / 365.25)
        future_df['day_of_year_cos'] = np.cos(2 * np.pi * future_df['day_of_year'] / 365.25)

        # ========================================================================
        # LAG FEATURES (look back into historical data)
        # ========================================================================
        # Combine historical sales with forecasted sales (updated iteratively)
        oos_adj_sales = cat_hist['oos_adjusted_sales'].values
        extended_sales = oos_adj_sales.copy()

        for lag in [7, 14, 28, 364]:
            lag_values = []
            for i in range(forecast_horizon):
                # Look back lag days from current forecast date
                lookback_idx = len(oos_adj_sales) + i - lag

                if lookback_idx >= 0 and lookback_idx < len(extended_sales):
                    lag_values.append(extended_sales[lookback_idx])
                elif lookback_idx >= 0:
                    # Use last known value if we've run out of extended sales
                    lag_values.append(extended_sales[-1])
                else:
                    lag_values.append(np.nan)

            future_df[f'lag_{lag}'] = lag_values

        # ========================================================================
        # ROLLING FEATURES (calculated from extended series)
        # ========================================================================
        # Similar approach - for MVP, use historical rolling means
        last_rolling_7 = cat_hist['rolling_mean_7'].iloc[-1]
        last_rolling_28 = cat_hist['rolling_mean_28'].iloc[-1]
        last_rolling_364 = cat_hist['rolling_mean_364'].iloc[-1]

        future_df['rolling_mean_7'] = last_rolling_7
        future_df['rolling_mean_28'] = last_rolling_28
        future_df['rolling_mean_364'] = last_rolling_364

        # ========================================================================
        # PROMOTION FEATURES
        # ========================================================================
        if future_promo_plan is not None and category in future_promo_plan['category'].values:
            # Merge with future promotion plan
            future_promo = future_promo_plan[future_promo_plan['category'] == category]
            future_df = future_df.merge(
                future_promo[['date', 'promo_intensity', 'promo_flag', 'avg_discount_pct']],
                on='date',
                how='left'
            )
        else:
            # Assume no promotions (MVP default)
            future_df['promo_intensity'] = 0.0
            future_df['promo_flag'] = 0
            future_df['avg_discount_pct'] = 0.0

        # Promo lag (look back 7 days)
        future_df['promo_lag_7'] = np.nan
        for i in range(forecast_horizon):
            lookback_idx = len(cat_hist) - 7 + i
            if lookback_idx >= 0 and lookback_idx < len(cat_hist):
                future_df.loc[i, 'promo_lag_7'] = cat_hist.iloc[lookback_idx]['promo_intensity']

        # Days since promo (simplified)
        last_promo_days = cat_hist['days_since_promo'].iloc[-1]
        future_df['days_since_promo'] = np.arange(last_promo_days + 1,
                                                   last_promo_days + forecast_horizon + 1)

        # ========================================================================
        # OOS FEATURES (assume minimal OOS in future)
        # ========================================================================
        future_df['oos_loss_share'] = 0.05  # 5% baseline
        future_df['oos_lag_7'] = cat_hist['oos_loss_share'].iloc[-7:].mean()

        # ========================================================================
        # HOLIDAY FEATURES (use holiday calendar)
        # ========================================================================
        # For MVP, replicate holiday detection logic
        # Simplified: check known holidays
        future_df['holiday_flag'] = 0
        future_df['days_to_holiday'] = 999
        future_df['days_from_holiday'] = 999
        future_df['is_pre_holiday'] = 0
        future_df['is_holiday_peak'] = 0
        future_df['is_post_holiday'] = 0

        # ========================================================================
        # WEATHER FEATURES
        # ========================================================================
        if future_weather is not None and category in future_weather['category'].values:
            # Merge with future weather forecast
            future_weather_cat = future_weather[future_weather['category'] == category]
            future_df = future_df.merge(
                future_weather_cat[['date', 'temperature', 'precipitation']],
                on='date',
                how='left'
            )
        else:
            # Use historical seasonal averages (MVP fallback)
            for i, date in enumerate(future_dates):
                day_of_year = date.dayofyear

                # Find similar days in historical data (±7 days)
                similar_days = cat_hist[
                    (cat_hist['date'].dt.dayofyear >= day_of_year - 7) &
                    (cat_hist['date'].dt.dayofyear <= day_of_year + 7)
                ]

                if len(similar_days) > 0:
                    future_df.loc[i, 'temperature'] = similar_days['temperature'].mean()
                    future_df.loc[i, 'precipitation'] = similar_days['precipitation'].mean()
                else:
                    future_df.loc[i, 'temperature'] = cat_hist['temperature'].mean()
                    future_df.loc[i, 'precipitation'] = 0.0

        # Weather derived features
        future_df['temp_rolling_avg_14'] = future_df['temperature'].rolling(14, min_periods=1).mean()
        future_df['temp_deviation'] = future_df['temperature'] - future_df['temp_rolling_avg_14']

        temp_10th = cat_hist['temperature'].quantile(0.1)
        temp_90th = cat_hist['temperature'].quantile(0.9)
        precip_90th = cat_hist['precipitation'].quantile(0.9)

        future_df['is_extreme_cold'] = (future_df['temperature'] < temp_10th).astype(int)
        future_df['is_extreme_hot'] = (future_df['temperature'] > temp_90th).astype(int)
        future_df['heavy_precip_flag'] = (future_df['precipitation'] > precip_90th).astype(int)

        # ========================================================================
        # ETS FEATURES
        # ========================================================================
        # Use last known ETS level
        last_ets_level = cat_hist['ets_level'].iloc[-1]
        future_df['ets_level'] = last_ets_level

        # Additional required features from original data
        future_df['base_price'] = cat_hist['base_price'].mean()
        future_df['final_price'] = cat_hist['final_price'].mean()

        return future_df

    def forecast_category(self,
                         historical_df: pd.DataFrame,
                         category: str,
                         forecast_horizon: int = 30,
                         **kwargs) -> pd.DataFrame:
        """
        Generate forecast for a category

        Args:
            historical_df: Historical data
            category: Category name
            forecast_horizon: Number of days to forecast
            **kwargs: Additional arguments for create_future_features

        Returns:
            Forecast DataFrame
        """
        print(f"\n  Generating forecast for {category}...")

        # Step 1: Generate ETS forecast
        ets_forecast = self.ets_decomposer.forecast(category, steps=forecast_horizon)

        # Step 2: Create future features
        future_features = self.create_future_features(
            historical_df, category, forecast_horizon, **kwargs
        )

        # Step 3: Predict residuals with LightGBM
        # Need to match feature columns exactly
        feature_cols = self.lgbm_model.models[category].feature_name()
        X_future = future_features[feature_cols].copy()
        X_future = X_future.fillna(method='ffill').fillna(method='bfill').fillna(0)

        lgbm_residual_forecast = self.lgbm_model.models[category].predict(
            X_future,
            num_iteration=self.lgbm_model.models[category].best_iteration
        )

        # Step 4: Combine forecasts
        final_forecast = ets_forecast + lgbm_residual_forecast

        # Step 5: Apply non-negativity constraint
        final_forecast = np.maximum(final_forecast, 0)

        # Create result DataFrame
        forecast_df = pd.DataFrame({
            'date': future_features['date'],
            'category': category,
            'ets_forecast': ets_forecast,
            'lgbm_residual_forecast': lgbm_residual_forecast,
            'final_forecast': final_forecast
        })

        print(f"    ETS forecast mean: {ets_forecast.mean():.2f}")
        print(f"    LGBM residual mean: {lgbm_residual_forecast.mean():.2f}")
        print(f"    Final forecast mean: {final_forecast.mean():.2f}")
        print(f"    ✓ Forecast generated")

        return forecast_df

    def forecast_all_categories(self,
                               historical_df: pd.DataFrame,
                               forecast_horizon: int = 30,
                               **kwargs) -> pd.DataFrame:
        """
        Generate forecasts for all categories

        Args:
            historical_df: Historical data
            forecast_horizon: Number of days to forecast
            **kwargs: Additional arguments for forecast_category

        Returns:
            Combined forecast DataFrame
        """
        print("="*60)
        print("GENERATING FORECASTS")
        print("="*60)
        print(f"Forecast horizon: {forecast_horizon} days")

        categories = historical_df['category'].unique()
        all_forecasts = []

        for category in categories:
            forecast_df = self.forecast_category(
                historical_df, category, forecast_horizon, **kwargs
            )
            all_forecasts.append(forecast_df)

        # Combine all forecasts
        combined_forecasts = pd.concat(all_forecasts, ignore_index=True)

        print("\n" + "="*60)
        print("FORECAST GENERATION COMPLETE")
        print("="*60)
        print(f"Categories forecasted: {len(categories)}")
        print(f"Total forecast records: {len(combined_forecasts)}")

        return combined_forecasts


if __name__ == "__main__":
    # Test forecast module
    print("Testing forecast module...")

    # Load required modules and data
    # (This would be run after training in main pipeline)
    pass
