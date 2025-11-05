"""
Feature Engineering Module

Creates ~40 features for category-day forecasting:
- Temporal features (weekly and annual seasonality)
- OOS-adjusted sales and OOS features
- Lag and rolling window features
- Promotion features
- Holiday features
- Weather features
- ETS-derived features (added later in pipeline)
"""

import pandas as pd
import numpy as np
from typing import Optional, List


class FeatureEngineer:
    """Feature engineering for category-level daily forecasting"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineer

        Args:
            df: Category-day level DataFrame
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.sort_values(['category', 'date'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Track feature creation
        self.features_created = []

    def create_all_features(self) -> pd.DataFrame:
        """
        Create all features in correct order

        Returns:
            DataFrame with all features
        """
        print("Creating all features...")
        print(f"Starting columns: {len(self.df.columns)}")

        # Order matters: OOS adjustment must come first
        self.create_oos_adjusted_sales()
        self.create_temporal_features()
        self.create_oos_features()
        self.create_lag_features()
        self.create_rolling_features()
        self.create_promotion_features()
        self.create_holiday_features()
        self.create_weather_features()

        print(f"\nFinal columns: {len(self.df.columns)}")
        print(f"Features created: {len(self.features_created)}")
        print(f"\nFeature summary by category:")
        for category in ['Temporal', 'OOS', 'Lags', 'Rolling', 'Promotion', 'Holiday', 'Weather']:
            cat_features = [f for f in self.features_created if category.lower() in f.lower() or
                          (category == 'Temporal' and any(x in f for x in ['day_', 'week_', 'month_', 'is_weekend'])) or
                          (category == 'Lags' and 'lag_' in f and 'oos' not in f and 'promo' not in f) or
                          (category == 'Rolling' and 'rolling' in f)]
            if cat_features:
                print(f"  {category}: {len(cat_features)} features")

        return self.df

    # ========================================================================
    # TEMPORAL FEATURES
    # ========================================================================

    def create_temporal_features(self) -> None:
        """Create temporal features for weekly and annual seasonality"""
        print("\n1. Creating temporal features...")

        df = self.df

        # Basic temporal features
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_month'] = ((df['date'].dt.day - 1) // 7 + 1)

        # Weekend flag
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Cyclical encoding - Weekly (day of week)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Cyclical encoding - Annual (week of year)
        df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

        # Cyclical encoding - Monthly
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Cyclical encoding - Day of year (fine-grained annual)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

        temporal_features = [
            'day_of_week', 'week_of_year', 'month', 'day_of_year', 'week_of_month',
            'is_weekend',
            'day_of_week_sin', 'day_of_week_cos',
            'week_of_year_sin', 'week_of_year_cos',
            'month_sin', 'month_cos',
            'day_of_year_sin', 'day_of_year_cos'
        ]

        self.features_created.extend(temporal_features)
        print(f"   Created {len(temporal_features)} temporal features")

    # ========================================================================
    # OOS FEATURES
    # ========================================================================

    def create_oos_adjusted_sales(self) -> None:
        """Create OOS-adjusted sales (true demand estimate)"""
        print("\n2. Creating OOS-adjusted sales...")

        # Clip OOS to prevent division by zero
        self.df['oos_loss_share'] = self.df['oos_loss_share'].clip(upper=0.99)

        # Calculate OOS-adjusted sales
        self.df['oos_adjusted_sales'] = self.df['sales'] / (1 - self.df['oos_loss_share'])

        self.features_created.append('oos_adjusted_sales')
        print(f"   Created OOS-adjusted sales")
        print(f"   Avg adjustment: {(self.df['oos_adjusted_sales'] / self.df['sales']).mean():.3f}x")

    def create_oos_features(self) -> None:
        """Create OOS-related features"""
        print("\n3. Creating OOS features...")

        # OOS lag (7 days)
        self.df['oos_lag_7'] = self.df.groupby('category')['oos_loss_share'].shift(7)

        oos_features = ['oos_lag_7']
        self.features_created.extend(oos_features)
        print(f"   Created {len(oos_features)} OOS features")

    # ========================================================================
    # LAG FEATURES
    # ========================================================================

    def create_lag_features(self) -> None:
        """Create lag features using OOS-adjusted sales"""
        print("\n4. Creating lag features...")

        lags = [7, 14, 28, 364]

        for lag in lags:
            col_name = f'lag_{lag}'
            self.df[col_name] = self.df.groupby('category')['oos_adjusted_sales'].shift(lag)
            self.features_created.append(col_name)

        print(f"   Created {len(lags)} lag features: {lags}")

    # ========================================================================
    # ROLLING WINDOW FEATURES
    # ========================================================================

    def create_rolling_features(self) -> None:
        """Create rolling window features using OOS-adjusted sales"""
        print("\n5. Creating rolling window features...")

        # Rolling mean 7 days
        self.df['rolling_mean_7'] = (
            self.df.groupby('category')['oos_adjusted_sales']
            .rolling(7, min_periods=7)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Rolling mean 28 days
        self.df['rolling_mean_28'] = (
            self.df.groupby('category')['oos_adjusted_sales']
            .rolling(28, min_periods=28)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Year-over-year rolling mean (7-day window around lag 364)
        shifted_364 = self.df.groupby('category')['oos_adjusted_sales'].shift(364)
        self.df['rolling_mean_364'] = (
            shifted_364
            .rolling(7, center=True, min_periods=7)
            .mean()
        )

        rolling_features = ['rolling_mean_7', 'rolling_mean_28', 'rolling_mean_364']
        self.features_created.extend(rolling_features)
        print(f"   Created {len(rolling_features)} rolling window features")

    # ========================================================================
    # PROMOTION FEATURES
    # ========================================================================

    def create_promotion_features(self) -> None:
        """Create promotion-related features"""
        print("\n6. Creating promotion features...")

        # Historical promotion lag
        self.df['promo_lag_7'] = self.df.groupby('category')['promo_intensity'].shift(7)

        # Days since last promotion
        self.df['days_since_promo'] = 0

        for category in self.df['category'].unique():
            mask = self.df['category'] == category
            cat_df = self.df[mask].copy()

            # Find promotion dates
            promo_dates = cat_df[cat_df['promo_flag'] == 1]['date'].values

            for idx in cat_df.index:
                current_date = self.df.loc[idx, 'date']

                # Find most recent promotion before current date
                past_promos = promo_dates[promo_dates < current_date]

                if len(past_promos) > 0:
                    days_since = (current_date - past_promos[-1]).days
                    self.df.loc[idx, 'days_since_promo'] = days_since
                else:
                    self.df.loc[idx, 'days_since_promo'] = 999  # No prior promo

        promo_features = ['promo_lag_7', 'days_since_promo']
        self.features_created.extend(promo_features)
        print(f"   Created {len(promo_features)} promotion features")

    # ========================================================================
    # HOLIDAY FEATURES
    # ========================================================================

    def create_holiday_features(self) -> None:
        """Create holiday-related features"""
        print("\n7. Creating holiday features...")

        # Create holiday phase indicators
        self.df['is_pre_holiday'] = (
            (self.df['days_to_holiday'] >= -7) &
            (self.df['days_to_holiday'] <= -1)
        ).astype(int)

        self.df['is_holiday_peak'] = (self.df['days_to_holiday'] == 0).astype(int)

        self.df['is_post_holiday'] = (
            (self.df['days_from_holiday'] >= 1) &
            (self.df['days_from_holiday'] <= 7)
        ).astype(int)

        holiday_features = ['is_pre_holiday', 'is_holiday_peak', 'is_post_holiday']
        self.features_created.extend(holiday_features)
        print(f"   Created {len(holiday_features)} holiday features")

    # ========================================================================
    # WEATHER FEATURES
    # ========================================================================

    def create_weather_features(self) -> None:
        """Create weather-related features"""
        print("\n8. Creating weather features...")

        # Temperature rolling average (14 days)
        self.df['temp_rolling_avg_14'] = (
            self.df.groupby('category')['temperature']
            .rolling(14, min_periods=14)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Temperature deviation from recent norm
        self.df['temp_deviation'] = self.df['temperature'] - self.df['temp_rolling_avg_14']

        # Extreme temperature flags (per category)
        self.df['is_extreme_cold'] = 0
        self.df['is_extreme_hot'] = 0
        self.df['heavy_precip_flag'] = 0

        for category in self.df['category'].unique():
            mask = self.df['category'] == category

            # Temperature extremes (10th and 90th percentiles)
            temp_10th = self.df.loc[mask, 'temperature'].quantile(0.1)
            temp_90th = self.df.loc[mask, 'temperature'].quantile(0.9)

            self.df.loc[mask, 'is_extreme_cold'] = (
                self.df.loc[mask, 'temperature'] < temp_10th
            ).astype(int)

            self.df.loc[mask, 'is_extreme_hot'] = (
                self.df.loc[mask, 'temperature'] > temp_90th
            ).astype(int)

            # Heavy precipitation (90th percentile)
            precip_90th = self.df.loc[mask, 'precipitation'].quantile(0.9)
            self.df.loc[mask, 'heavy_precip_flag'] = (
                self.df.loc[mask, 'precipitation'] > precip_90th
            ).astype(int)

        weather_features = [
            'temp_rolling_avg_14', 'temp_deviation',
            'is_extreme_cold', 'is_extreme_hot', 'heavy_precip_flag'
        ]
        self.features_created.extend(weather_features)
        print(f"   Created {len(weather_features)} weather features")

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_feature_list(self, exclude_target: bool = True) -> List[str]:
        """
        Get list of engineered features

        Args:
            exclude_target: Exclude sales and oos_adjusted_sales

        Returns:
            List of feature column names
        """
        exclude_cols = [
            'date', 'category', 'sales', 'oos_adjusted_sales',
            'sku_id'  # In case it exists
        ]

        if exclude_target:
            feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        else:
            feature_cols = [col for col in self.df.columns
                          if col not in ['date', 'category', 'sku_id']]

        return feature_cols

    def get_feature_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for all features

        Returns:
            Summary DataFrame
        """
        feature_cols = self.get_feature_list(exclude_target=False)
        return self.df[feature_cols].describe()

    def check_missing_values(self) -> pd.Series:
        """
        Check missing values in features

        Returns:
            Series with missing value counts
        """
        missing = self.df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) == 0:
            print("✓ No missing values in features")
        else:
            print(f"⚠ Missing values found in {len(missing)} features:")
            print(missing)

        return missing


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to engineer all features

    Args:
        df: Category-day DataFrame

    Returns:
        DataFrame with all engineered features
    """
    engineer = FeatureEngineer(df)
    df_features = engineer.create_all_features()

    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Total features created: {len(engineer.features_created)}")
    print(f"Output shape: {df_features.shape}")

    # Check for missing values
    print("\nChecking for missing values...")
    engineer.check_missing_values()

    return df_features


if __name__ == "__main__":
    # Test feature engineering
    print("Testing feature engineering module...")

    # Load category data
    category_df = pd.read_csv("data/category_day_data.csv")
    category_df['date'] = pd.to_datetime(category_df['date'])

    # Engineer features
    df_with_features = engineer_features(category_df)

    # Display sample
    print("\nSample data with features (first 5 rows):")
    print(df_with_features.head())

    # Save
    output_path = "data/category_day_features.csv"
    df_with_features.to_csv(output_path, index=False)
    print(f"\nData with features saved to: {output_path}")
