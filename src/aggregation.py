"""
Data Aggregation Module

Aggregates SKU-day level data to Category-day level with instock filtering.
Removes low-instock records and fills gaps with interpolation.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import AGGREGATION_CONFIG


def filter_low_instock(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    """
    Filter out SKU-day records where instock rate is below threshold

    Args:
        df: SKU-day level DataFrame
        threshold: Minimum instock rate (%). If None, uses config value

    Returns:
        Filtered DataFrame
    """
    if threshold is None:
        threshold = AGGREGATION_CONFIG['instock_threshold']

    initial_count = len(df)
    df_filtered = df[df['instock_rate'] >= threshold].copy()
    filtered_count = len(df_filtered)
    removed_count = initial_count - filtered_count
    removed_pct = (removed_count / initial_count * 100) if initial_count > 0 else 0

    print(f"\n  Instock filtering (threshold: {threshold}%):")
    print(f"    Initial records: {initial_count:,}")
    print(f"    Filtered records: {filtered_count:,}")
    print(f"    Removed: {removed_count:,} ({removed_pct:.1f}%)")

    return df_filtered


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

    # Step 1: Filter low instock records
    df_filtered = filter_low_instock(df)

    # Step 2: Calculate sales under promotion
    df_filtered = df_filtered.copy()
    df_filtered['sales_on_promo'] = df_filtered['sales'] * df_filtered['high_impact_promo']

    # Step 3: Group by category and date
    grouper = df_filtered.groupby(['category', 'date'])

    # Step 4: Aggregation
    agg_dict = {
        # Sales - Sum
        'sales': 'sum',

        # Instock rate - Volume-weighted average
        'instock_rate': lambda x: np.average(
            x, weights=df_filtered.loc[x.index, 'sales']
        ) if df_filtered.loc[x.index, 'sales'].sum() > 0 else x.mean(),

        # Sales on promo - Sum (for intensity calculation)
        'sales_on_promo': 'sum',

        # High impact promo - Count SKUs on promo
        'high_impact_promo': 'sum',

        # Price - Volume-weighted average
        'final_sku_price': lambda x: np.average(
            x, weights=df_filtered.loc[x.index, 'sales']
        ) if df_filtered.loc[x.index, 'sales'].sum() > 0 else x.mean(),

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

    # Step 8: Fill gaps with interpolation
    category_df = fill_gaps_with_interpolation(category_df)

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


def fill_gaps_with_interpolation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing dates with interpolation after instock filtering

    For each category:
    - Create complete date range
    - Interpolate numeric columns for missing dates

    Args:
        df: Category-day DataFrame (may have gaps)

    Returns:
        DataFrame with filled gaps
    """
    method = AGGREGATION_CONFIG['interpolation_method']
    limit = AGGREGATION_CONFIG['interpolation_limit']

    print(f"\n  Filling gaps with {method} interpolation (limit: {limit} days)...")

    categories = df['category'].unique()
    filled_dfs = []
    total_gaps_filled = 0

    for category in categories:
        cat_df = df[df['category'] == category].copy()

        # Create complete date range
        min_date = cat_df['date'].min()
        max_date = cat_df['date'].max()
        complete_dates = pd.date_range(start=min_date, end=max_date, freq='D')

        # Check for gaps
        existing_dates = set(cat_df['date'])
        missing_dates = set(complete_dates) - existing_dates
        gaps_count = len(missing_dates)

        if gaps_count > 0:
            print(f"    {category}: {gaps_count} missing dates")
            total_gaps_filled += gaps_count

            # Create complete dataframe
            complete_df = pd.DataFrame({'date': complete_dates, 'category': category})
            complete_df = complete_df.merge(cat_df, on=['date', 'category'], how='left')

            # Interpolate numeric columns
            numeric_cols = complete_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                complete_df[col] = complete_df[col].interpolate(
                    method=method,
                    limit=limit,
                    limit_direction='both'
                )

            # Forward fill any remaining NaNs
            complete_df.fillna(method='ffill', inplace=True)
            complete_df.fillna(method='bfill', inplace=True)

            filled_dfs.append(complete_df)
        else:
            filled_dfs.append(cat_df)

    result_df = pd.concat(filled_dfs, ignore_index=True)
    result_df.sort_values(['category', 'date'], inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    if total_gaps_filled > 0:
        print(f"    Total gaps filled: {total_gaps_filled}")
    else:
        print(f"    No gaps found - complete date coverage")

    return result_df


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
