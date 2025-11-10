"""
Data Aggregation Module

Aggregates SKU-day level data to Category-day level with SKU-level interpolation.
Interpolates low-instock SKU records before aggregation to preserve SKU-specific patterns.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import AGGREGATION_CONFIG


def filter_low_instock(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    """
    Interpolate SKU-day records where instock rate is below threshold

    For each SKU:
    - Interior gaps (low instock between good days): Linear interpolation
    - Trailing gaps (low instock till end): Forward-fill from last good value

    Args:
        df: SKU-day level DataFrame
        threshold: Minimum instock rate (%). If None, uses config value

    Returns:
        DataFrame with interpolated values for low-instock periods
    """
    if threshold is None:
        threshold = AGGREGATION_CONFIG['instock_threshold']

    method = AGGREGATION_CONFIG['interpolation_method']
    limit = AGGREGATION_CONFIG['interpolation_limit']

    initial_count = len(df)
    low_instock_count = (df['instock_rate'] < threshold).sum()

    print(f"\n  SKU-level interpolation for low instock (threshold: {threshold}%):")
    print(f"    Total records: {initial_count:,}")
    print(f"    Records below threshold: {low_instock_count:,} ({low_instock_count/initial_count*100:.1f}%)")

    if low_instock_count == 0:
        print(f"    No interpolation needed")
        return df

    df_interpolated = df.copy()
    skus = df_interpolated['sku_id'].unique()
    interpolated_count = 0
    forward_filled_count = 0

    for sku in skus:
        sku_mask = df_interpolated['sku_id'] == sku
        sku_data = df_interpolated[sku_mask].copy()
        sku_data = sku_data.sort_values('date').reset_index(drop=True)

        # Identify low instock records
        low_instock_mask = sku_data['instock_rate'] < threshold

        if not low_instock_mask.any():
            continue

        # Mark values to interpolate
        numeric_cols = ['sales', 'instock_rate', 'final_sku_price']
        for col in numeric_cols:
            if col in sku_data.columns:
                # Create a copy with NaN for low instock values
                values = sku_data[col].copy()
                values[low_instock_mask] = np.nan

                # Interpolate (handles interior gaps)
                interpolated = values.interpolate(
                    method=method,
                    limit=limit,
                    limit_direction='both'
                )

                # Track what was interpolated vs forward-filled
                was_nan_before_interp = values.isna()
                still_nan_after_interp = interpolated.isna()

                # Forward fill remaining NaNs (trailing gaps)
                final_values = interpolated.ffill()

                # Update the dataframe
                df_interpolated.loc[sku_mask, col] = final_values.values

                # Count interpolations
                if col == 'sales':  # Count once per row
                    interpolated_count += (was_nan_before_interp & ~still_nan_after_interp).sum()
                    forward_filled_count += (still_nan_after_interp & ~final_values.isna()).sum()

        # Update instock_rate to threshold for interpolated records (mark as "acceptable")
        df_interpolated.loc[sku_mask & (df_interpolated['instock_rate'] < threshold), 'instock_rate'] = threshold

    print(f"    Interpolated (interior gaps): {interpolated_count:,}")
    print(f"    Forward-filled (trailing gaps): {forward_filled_count:,}")

    return df_interpolated


def aggregate_to_category(df: pd.DataFrame,
                         weather_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Aggregate SKU-day data to Category-day level

    Expected SKU-day columns:
        - date, sku_id, category
        - sales
        - instock_rate (0-100%)
        - high_impact_promo (0/1)
        - final_sku_price

    Expected weather_df columns (category-day level):
        - date, category
        - avg_temperature, avg_rainfall, avg_snowfall
        - holiday_flag (14 to -5 for holiday period, -100 otherwise)

    Process:
        1. Interpolate low-instock SKU-day records (preserves SKU patterns)
        2. Aggregate to category-day level

    Aggregation rules:
        - Sales: Sum across SKUs
        - Instock rate: Volume-weighted average
        - Promo: Calculate intensity (% of sales on promo)
        - Price: Volume-weighted average
        - Weather/Holiday: Merge from category-level data

    Args:
        df: SKU-day level DataFrame
        weather_df: Category-day level weather/holiday data (optional)

    Returns:
        Category-day level DataFrame
    """
    print("="*60)
    print("AGGREGATING SKU-DAY DATA TO CATEGORY-DAY LEVEL")
    print("="*60)
    print(f"  Input: {len(df):,} SKU-day records")
    print(f"  Input date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Categories: {df['category'].nunique()}")
    print(f"  SKUs: {df['sku_id'].nunique()}")

    # Step 1: Interpolate low instock records at SKU level
    df_interpolated = filter_low_instock(df)

    # Step 2: Calculate sales under promotion
    df_interpolated = df_interpolated.copy()
    df_interpolated['sales_on_promo'] = df_interpolated['sales'] * df_interpolated['high_impact_promo']

    # Step 3: Group by category and date
    grouper = df_interpolated.groupby(['category', 'date'])

    # Step 4: Aggregation
    agg_dict = {
        # Sales - Sum
        'sales': 'sum',

        # Instock rate - Volume-weighted average
        'instock_rate': lambda x: np.average(
            x, weights=df_interpolated.loc[x.index, 'sales']
        ) if df_interpolated.loc[x.index, 'sales'].sum() > 0 else x.mean(),

        # Sales on promo - Sum (for intensity calculation)
        'sales_on_promo': 'sum',

        # High impact promo - Count SKUs on promo
        'high_impact_promo': 'sum',

        # Price - Volume-weighted average
        'final_sku_price': lambda x: np.average(
            x, weights=df_interpolated.loc[x.index, 'sales']
        ) if df_interpolated.loc[x.index, 'sales'].sum() > 0 else x.mean(),

        # Count of SKUs
        'sku_id': 'count'
    }

    category_df = grouper.agg(agg_dict).reset_index()
    category_df.rename(columns={'sku_id': 'num_skus'}, inplace=True)

    # Step 5: Calculate derived metrics
    # Promo intensity (% of sales on high-impact promo)
    category_df['promo_intensity'] = np.where(
        category_df['sales'] > 0,
        category_df['sales_on_promo'] / category_df['sales'],
        0
    )

    # Binary promo flag (>20% of sales on promo)
    category_df['promo_flag'] = (category_df['promo_intensity'] > 0.2).astype(int)

    # Percentage of SKUs on high impact promo (normalized feature)
    category_df['pct_skus_on_high_impact_promo'] = np.where(
        category_df['num_skus'] > 0,
        category_df['high_impact_promo'] / category_df['num_skus'],
        0
    )

    # Drop intermediate columns (keep num_skus for now but won't be used as feature)
    category_df.drop(columns=['sales_on_promo'], inplace=True)

    # Step 6: Merge weather and holiday data
    if weather_df is not None:
        print(f"\n  Merging weather/holiday data...")
        category_df = category_df.merge(
            weather_df[['date', 'category', 'avg_temperature', 'avg_rainfall',
                       'avg_snowfall', 'holiday_flag']],
            on=['date', 'category'],
            how='left'
        )
        print(f"    Weather data merged")

    # Step 7: Sort by category and date
    category_df.sort_values(['category', 'date'], inplace=True)
    category_df.reset_index(drop=True, inplace=True)

    print(f"\n  Output: {len(category_df):,} Category-day records")
    print(f"  Date range: {category_df['date'].min()} to {category_df['date'].max()}")

    # Display summary
    print("\n  Aggregated data summary by category:")
    summary = category_df.groupby('category').agg({
        'sales': ['mean', 'std', 'min', 'max'],
        'instock_rate': 'mean',
        'promo_intensity': 'mean',
        'pct_skus_on_high_impact_promo': 'mean',
        'num_skus': 'mean'
    }).round(2)
    print(summary)

    print("\n" + "="*60)
    print("AGGREGATION COMPLETE")
    print("="*60)

    return category_df


def validate_aggregated_data(df: pd.DataFrame) -> bool:
    """
    Validate aggregated category-day data

    Args:
        df: Category-day DataFrame

    Returns:
        True if validation passes
    """
    print("\n" + "="*60)
    print("VALIDATING AGGREGATED DATA")
    print("="*60)

    checks_passed = True

    # Check for required columns
    required_cols = [
        'category', 'date', 'sales', 'instock_rate',
        'promo_intensity', 'promo_flag', 'final_sku_price', 'num_skus'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  ✗ Missing columns: {missing_cols}")
        checks_passed = False
    else:
        print(f"  ✓ All required columns present")

    # Check for missing values in key columns
    key_cols = ['category', 'date', 'sales']
    for col in key_cols:
        if df[col].isna().any():
            print(f"  ✗ Missing values in {col}")
            checks_passed = False
    if not any(df[col].isna().any() for col in key_cols):
        print(f"  ✓ No missing values in key columns")

    # Check for negative sales
    if (df['sales'] < 0).any():
        print(f"  ✗ Negative sales found")
        checks_passed = False
    else:
        print(f"  ✓ No negative sales")

    # Check instock rate in valid range [0, 100]
    if not df['instock_rate'].between(0, 100).all():
        print(f"  ✗ Instock rate outside [0, 100] range")
        checks_passed = False
    else:
        print(f"  ✓ Instock rate in valid range")

    # Check promo intensity in valid range [0, 1]
    if not df['promo_intensity'].between(0, 1).all():
        print(f"  ✗ Promo intensity outside [0, 1] range")
        checks_passed = False
    else:
        print(f"  ✓ Promo intensity in valid range")

    # Check date continuity for each category
    print("\n  Date continuity check:")
    for category in df['category'].unique():
        cat_dates = df[df['category'] == category]['date'].sort_values()
        expected_days = (cat_dates.max() - cat_dates.min()).days + 1
        actual_days = len(cat_dates)

        if expected_days != actual_days:
            print(f"    ⚠ {category}: Date gaps detected ({actual_days}/{expected_days} days)")
        else:
            print(f"    ✓ {category}: Complete date sequence")

    if checks_passed:
        print("\n✓ All validation checks passed!")
    else:
        print("\n✗ Some validation checks failed!")

    print("="*60)

    return checks_passed
