"""
Data Aggregation Module

Aggregates SKU-day level data to Category-day level following
volume-weighted averaging for all metrics.
"""

import pandas as pd
import numpy as np
from typing import Optional


def aggregate_to_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SKU-day data to Category-day level

    Aggregation rules:
    - Sales: Sum
    - OOS: Volume-weighted average
    - Promotion: Calculate intensity and weighted metrics
    - Price: Volume-weighted average
    - Weather: Mean (assumes same geography)
    - Holiday: Max for flags, first for continuous vars

    Args:
        df: SKU-day level DataFrame

    Returns:
        Category-day level DataFrame
    """
    print("Aggregating SKU-day data to Category-day level...")
    print(f"  Input: {len(df):,} SKU-day records")
    print(f"  Input date range: {df['date'].min()} to {df['date'].max()}")

    # Make a copy to avoid modifying original
    df = df.copy()

    # Calculate sales under promotion for intensity calculation
    df['sales_on_promo'] = df['sales'] * df['promo_flag']

    # Calculate discount percentage
    df['discount_pct'] = np.where(
        df['base_price'] > 0,
        (df['base_price'] - df['final_price']) / df['base_price'],
        0
    )

    # Group by category and date
    grouper = df.groupby(['category', 'date'])

    # Aggregation operations
    agg_dict = {
        # Sales - Sum
        'sales': 'sum',

        # OOS - Volume-weighted average
        'oos_loss_share': lambda x: np.average(
            x, weights=df.loc[x.index, 'sales']
        ) if df.loc[x.index, 'sales'].sum() > 0 else 0,

        # Sales on promo - Sum (for intensity calculation)
        'sales_on_promo': 'sum',

        # Discount - Volume-weighted average
        'discount_pct': lambda x: np.average(
            x, weights=df.loc[x.index, 'sales']
        ) if df.loc[x.index, 'sales'].sum() > 0 else 0,

        # Price - Volume-weighted average
        'base_price': lambda x: np.average(
            x, weights=df.loc[x.index, 'sales']
        ) if df.loc[x.index, 'sales'].sum() > 0 else 0,

        'final_price': lambda x: np.average(
            x, weights=df.loc[x.index, 'sales']
        ) if df.loc[x.index, 'sales'].sum() > 0 else 0,

        # Weather - Mean (same geography)
        'temperature': 'mean',
        'precipitation': 'mean',

        # Holiday - Max for flags, first for continuous
        'holiday_flag': 'max',
        'days_to_holiday': 'first',
        'days_from_holiday': 'first',
        'holiday_name': 'first'
    }

    # Perform aggregation
    category_df = grouper.agg(agg_dict).reset_index()

    # Calculate promotion intensity
    category_df['promo_intensity'] = np.where(
        category_df['sales'] > 0,
        category_df['sales_on_promo'] / category_df['sales'],
        0
    )

    # Create binary promotion flag (>20% of sales on promotion)
    category_df['promo_flag'] = (category_df['promo_intensity'] > 0.2).astype(int)

    # Rename discount_pct to avg_discount_pct for clarity
    category_df.rename(columns={'discount_pct': 'avg_discount_pct'}, inplace=True)

    # Drop intermediate column
    category_df.drop(columns=['sales_on_promo'], inplace=True)

    # Sort by category and date
    category_df.sort_values(['category', 'date'], inplace=True)
    category_df.reset_index(drop=True, inplace=True)

    print(f"  Output: {len(category_df):,} Category-day records")
    print(f"  Categories: {category_df['category'].nunique()}")
    print(f"  Days per category: {len(category_df) // category_df['category'].nunique()}")

    # Display aggregation summary
    print("\nAggregated data summary by category:")
    summary = category_df.groupby('category').agg({
        'sales': ['mean', 'std', 'min', 'max'],
        'promo_intensity': 'mean',
        'oos_loss_share': 'mean',
        'avg_discount_pct': 'mean'
    }).round(2)
    print(summary)

    return category_df


def validate_aggregated_data(df: pd.DataFrame) -> bool:
    """
    Validate aggregated category-day data

    Args:
        df: Category-day DataFrame

    Returns:
        True if validation passes
    """
    print("\nValidating aggregated data...")

    checks_passed = True

    # Check for required columns
    required_cols = [
        'category', 'date', 'sales', 'oos_loss_share',
        'promo_intensity', 'promo_flag', 'avg_discount_pct',
        'temperature', 'precipitation', 'holiday_flag',
        'days_to_holiday', 'days_from_holiday', 'holiday_name',
        'base_price', 'final_price'
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

    # Check OOS in valid range [0, 1]
    if not df['oos_loss_share'].between(0, 1).all():
        print(f"  ✗ OOS loss share outside [0, 1] range")
        checks_passed = False
    else:
        print(f"  ✓ OOS loss share in valid range")

    # Check promo intensity in valid range [0, 1]
    if not df['promo_intensity'].between(0, 1).all():
        print(f"  ✗ Promo intensity outside [0, 1] range")
        checks_passed = False
    else:
        print(f"  ✓ Promo intensity in valid range")

    # Check date continuity for each category
    for category in df['category'].unique():
        cat_dates = df[df['category'] == category]['date'].sort_values()
        expected_days = (cat_dates.max() - cat_dates.min()).days + 1
        actual_days = len(cat_dates)

        if expected_days != actual_days:
            print(f"  ⚠ {category}: Date gaps detected ({actual_days}/{expected_days} days)")
        else:
            print(f"  ✓ {category}: Complete date sequence")

    if checks_passed:
        print("\n✓ All validation checks passed!")
    else:
        print("\n✗ Some validation checks failed!")

    return checks_passed


def save_aggregated_data(df: pd.DataFrame,
                        output_path: str = "data/category_day_data.csv") -> None:
    """
    Save aggregated category-day data

    Args:
        df: Category-day DataFrame
        output_path: Output file path
    """
    df.to_csv(output_path, index=False)
    print(f"\nAggregated data saved to: {output_path}")
    print(f"File size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


if __name__ == "__main__":
    # Test aggregation
    print("Testing aggregation module...")

    # Load SKU data
    sku_df = pd.read_csv("data/sku_day_data.csv")
    sku_df['date'] = pd.to_datetime(sku_df['date'])

    # Aggregate
    category_df = aggregate_to_category(sku_df)

    # Validate
    validate_aggregated_data(category_df)

    # Save
    save_aggregated_data(category_df)

    # Display sample
    print("\nSample aggregated data (first 10 rows):")
    print(category_df.head(10))
