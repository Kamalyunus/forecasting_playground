"""
Synthetic Data Generator for Category-Level Forecasting

Generates realistic SKU-day level sales data with:
- Weekly and annual seasonality
- Promotions and price variations
- Out-of-stock events
- Weather effects
- Holiday impacts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


class SyntheticDataGenerator:
    """Generate synthetic SKU-day level sales data for forecasting MVP"""

    def __init__(self,
                 start_date: str = "2022-01-01",
                 n_days: int = 730,
                 categories: List[str] = None,
                 skus_per_category: int = 10,
                 seed: int = 42):
        """
        Initialize data generator

        Args:
            start_date: Start date for data generation
            n_days: Number of days to generate (default 730 = 2 years)
            categories: List of category names (default: 3 categories)
            skus_per_category: Number of SKUs per category
            seed: Random seed for reproducibility
        """
        self.start_date = pd.to_datetime(start_date)
        self.n_days = n_days
        self.categories = categories or ["Beverages", "Frozen_Foods", "Bakery"]
        self.skus_per_category = skus_per_category
        self.seed = seed

        np.random.seed(seed)

        # Generate date range
        self.dates = pd.date_range(start=self.start_date, periods=n_days, freq='D')

        # Define holidays (simplified for MVP)
        self.holidays = self._define_holidays()

    def _define_holidays(self) -> Dict[str, List[pd.Timestamp]]:
        """Define major holidays for 2 years"""
        holidays = {
            'Thanksgiving': [
                pd.Timestamp('2022-11-24'),
                pd.Timestamp('2023-11-23')
            ],
            'Christmas': [
                pd.Timestamp('2022-12-25'),
                pd.Timestamp('2023-12-25')
            ],
            'New_Year': [
                pd.Timestamp('2022-01-01'),
                pd.Timestamp('2023-01-01')
            ],
            'July_4th': [
                pd.Timestamp('2022-07-04'),
                pd.Timestamp('2023-07-04')
            ],
            'Memorial_Day': [
                pd.Timestamp('2022-05-30'),
                pd.Timestamp('2023-05-29')
            ],
            'Labor_Day': [
                pd.Timestamp('2022-09-05'),
                pd.Timestamp('2023-09-04')
            ]
        }
        return holidays

    def _get_holiday_info(self, date: pd.Timestamp) -> Tuple[int, str, int, int]:
        """
        Get holiday information for a given date

        Returns:
            holiday_flag, holiday_name, days_to_holiday, days_from_holiday
        """
        min_days_to = 999
        min_days_from = 999
        closest_holiday = None

        for holiday_name, holiday_dates in self.holidays.items():
            for holiday_date in holiday_dates:
                days_diff = (holiday_date - date).days

                if days_diff == 0:
                    return 1, holiday_name, 0, 0
                elif days_diff > 0 and days_diff < abs(min_days_to):
                    min_days_to = -days_diff
                    closest_holiday = holiday_name
                elif days_diff < 0 and abs(days_diff) < abs(min_days_from):
                    min_days_from = abs(days_diff)
                    if closest_holiday is None:
                        closest_holiday = holiday_name

        # Within 7 days before or after holiday
        if abs(min_days_to) <= 7 or abs(min_days_from) <= 7:
            holiday_flag = 1
        else:
            holiday_flag = 0

        return holiday_flag, closest_holiday or 'None', min_days_to, min_days_from

    def _generate_base_demand(self,
                              date: pd.Timestamp,
                              category: str,
                              sku_base_level: float) -> float:
        """
        Generate base demand with weekly and annual seasonality

        Args:
            date: Date for demand
            category: Category name
            sku_base_level: Base sales level for SKU

        Returns:
            Base demand value
        """
        # Day of week effect (weekend higher for retail)
        day_of_week = date.dayofweek
        weekend_lift = 1.3 if day_of_week >= 5 else 1.0
        weekday_pattern = {0: 0.9, 1: 0.85, 2: 0.88, 3: 0.92, 4: 1.1, 5: 1.3, 6: 1.25}
        dow_factor = weekday_pattern.get(day_of_week, 1.0)

        # Annual seasonality (weather-driven patterns per category)
        day_of_year = date.dayofyear

        # Category-specific annual patterns
        if category == "Beverages":
            # Peak in summer (hot weather)
            annual_factor = 1.0 + 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
        elif category == "Frozen_Foods":
            # Higher in summer (ice cream) and winter (frozen meals)
            annual_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25) + \
                           0.15 * np.sin(4 * np.pi * day_of_year / 365.25)
        else:  # Bakery
            # More stable, slight peak around holidays
            annual_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (day_of_year - 100) / 365.25)

        # Combine factors
        base_demand = sku_base_level * dow_factor * annual_factor

        return base_demand

    def _generate_weather(self, date: pd.Timestamp) -> Tuple[float, float]:
        """
        Generate synthetic weather data

        Returns:
            temperature (°F), precipitation (inches)
        """
        day_of_year = date.dayofyear

        # Temperature: seasonal pattern + daily variation
        # Annual average ~60°F, range 30-90°F
        seasonal_temp = 60 + 25 * np.sin(2 * np.pi * (day_of_year - 100) / 365.25)
        daily_variation = np.random.normal(0, 8)
        temperature = np.clip(seasonal_temp + daily_variation, 20, 100)

        # Precipitation: More in spring/fall, less in summer/winter
        seasonal_precip_prob = 0.2 + 0.15 * np.sin(2 * np.pi * (day_of_year - 50) / 365.25)
        if np.random.random() < seasonal_precip_prob:
            precipitation = np.random.gamma(2, 0.2)  # Exponential-like distribution
        else:
            precipitation = 0.0

        return round(temperature, 1), round(precipitation, 2)

    def _generate_promotions(self, date: pd.Timestamp, sku_id: str) -> Tuple[int, str, float]:
        """
        Generate promotion information

        Returns:
            promo_flag, promo_type, discount_pct
        """
        # Promotions more likely on weekends and around holidays
        day_of_week = date.dayofweek
        is_weekend = day_of_week >= 5

        # Check if near holiday
        holiday_flag, _, days_to_holiday, days_from_holiday = self._get_holiday_info(date)
        is_near_holiday = abs(days_to_holiday) <= 7 or abs(days_from_holiday) <= 3

        # Base promotion probability
        base_promo_prob = 0.1
        if is_weekend:
            base_promo_prob += 0.1
        if is_near_holiday:
            base_promo_prob += 0.15

        # Hash SKU to create consistent but varied promo patterns
        sku_hash = hash(sku_id) % 100
        promo_prob = base_promo_prob * (0.5 + sku_hash / 100.0)

        if np.random.random() < promo_prob:
            promo_flag = 1
            promo_types = ['Discount', 'BOGO', 'Bundle', 'Loyalty']
            promo_type = np.random.choice(promo_types, p=[0.5, 0.2, 0.2, 0.1])

            # Discount varies by type
            if promo_type == 'Discount':
                discount_pct = np.random.uniform(0.1, 0.4)
            elif promo_type == 'BOGO':
                discount_pct = 0.5  # Effective 50% off
            elif promo_type == 'Bundle':
                discount_pct = np.random.uniform(0.15, 0.3)
            else:  # Loyalty
                discount_pct = np.random.uniform(0.05, 0.2)
        else:
            promo_flag = 0
            promo_type = 'None'
            discount_pct = 0.0

        return promo_flag, promo_type, round(discount_pct, 3)

    def _generate_oos(self, date: pd.Timestamp, sku_id: str) -> float:
        """
        Generate out-of-stock loss share

        Returns:
            oos_loss_share (0-1)
        """
        # OOS more likely during high demand periods
        day_of_week = date.dayofweek
        is_weekend = day_of_week >= 5

        # Base OOS probability
        base_oos_prob = 0.05
        if is_weekend:
            base_oos_prob += 0.03

        # Check holiday proximity
        holiday_flag, _, days_to_holiday, _ = self._get_holiday_info(date)
        if -3 <= days_to_holiday <= -1:
            base_oos_prob += 0.08

        # Random OOS events
        if np.random.random() < base_oos_prob:
            # OOS severity varies
            oos_loss_share = np.random.beta(2, 5)  # Most OOS are partial
        else:
            oos_loss_share = 0.0

        return round(oos_loss_share, 3)

    def generate_sku_day_data(self) -> pd.DataFrame:
        """
        Generate complete SKU-day level dataset

        Returns:
            DataFrame with all SKU-day records
        """
        print("Generating synthetic SKU-day level data...")
        print(f"  Categories: {len(self.categories)}")
        print(f"  SKUs per category: {self.skus_per_category}")
        print(f"  Date range: {self.start_date.date()} to {self.dates[-1].date()}")
        print(f"  Total days: {self.n_days}")

        records = []

        # Generate base prices and levels for each SKU
        sku_info = {}
        for category in self.categories:
            # Category base price range
            if category == "Beverages":
                price_range = (2.0, 8.0)
                volume_range = (800, 1500)
            elif category == "Frozen_Foods":
                price_range = (3.0, 12.0)
                volume_range = (500, 1200)
            else:  # Bakery
                price_range = (2.5, 10.0)
                volume_range = (600, 1300)

            for i in range(self.skus_per_category):
                sku_id = f"{category}_SKU_{i+1:03d}"
                base_price = np.random.uniform(*price_range)
                base_volume = np.random.uniform(*volume_range)
                sku_info[sku_id] = {
                    'category': category,
                    'base_price': round(base_price, 2),
                    'base_volume': base_volume
                }

        total_skus = len(sku_info)
        print(f"  Total SKUs: {total_skus}")
        print("\nGenerating daily records...")

        # Generate data for each SKU and date
        for date in self.dates:
            # Generate weather once per day
            temperature, precipitation = self._generate_weather(date)

            # Get holiday info once per day
            holiday_flag, holiday_name, days_to_holiday, days_from_holiday = \
                self._get_holiday_info(date)

            for sku_id, info in sku_info.items():
                category = info['category']
                base_price = info['base_price']
                base_volume = info['base_volume']

                # Generate base demand
                base_demand = self._generate_base_demand(date, category, base_volume)

                # Generate promotion
                promo_flag, promo_type, discount_pct = self._generate_promotions(date, sku_id)

                # Promotion lift
                if promo_flag:
                    promo_lift = 1.0 + (discount_pct * np.random.uniform(2.0, 3.5))
                else:
                    promo_lift = 1.0

                # Holiday effect
                if holiday_flag:
                    if -7 <= days_to_holiday <= -1:
                        holiday_lift = 1.3 + (abs(days_to_holiday) / 7) * 0.2
                    elif days_to_holiday == 0:
                        holiday_lift = 1.5
                    else:  # Post-holiday dip
                        holiday_lift = 0.7
                else:
                    holiday_lift = 1.0

                # Weather effect (category-specific)
                if category == "Beverages":
                    # Hot weather increases beverages
                    weather_lift = 1.0 + max(0, (temperature - 75) / 100.0)
                elif category == "Frozen_Foods":
                    # Extreme temps increase frozen
                    weather_lift = 1.0 + abs(temperature - 60) / 200.0
                else:
                    weather_lift = 1.0

                # Rain effect (slight negative)
                if precipitation > 0.5:
                    rain_factor = 0.92
                else:
                    rain_factor = 1.0

                # Calculate true demand
                true_demand = base_demand * promo_lift * holiday_lift * weather_lift * rain_factor

                # Add noise
                noise_factor = np.random.normal(1.0, 0.1)
                true_demand *= noise_factor
                true_demand = max(0, true_demand)

                # Generate OOS
                oos_loss_share = self._generate_oos(date, sku_id)

                # Observed sales (reduced by OOS)
                sales = true_demand * (1 - oos_loss_share)

                # Calculate prices
                final_price = base_price * (1 - discount_pct)

                # Create record
                record = {
                    'date': date,
                    'sku_id': sku_id,
                    'category': category,
                    'sales': round(sales, 2),
                    'oos_loss_share': oos_loss_share,
                    'promo_flag': promo_flag,
                    'promo_type': promo_type,
                    'base_price': base_price,
                    'final_price': round(final_price, 2),
                    'temperature': temperature,
                    'precipitation': precipitation,
                    'holiday_flag': holiday_flag,
                    'days_to_holiday': days_to_holiday,
                    'days_from_holiday': days_from_holiday,
                    'holiday_name': holiday_name
                }

                records.append(record)

        # Create DataFrame
        df = pd.DataFrame(records)

        print(f"\nGenerated {len(df):,} records")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  Categories: {df['category'].nunique()}")
        print(f"  SKUs: {df['sku_id'].nunique()}")
        print(f"  Avg daily sales per SKU: {df.groupby('sku_id')['sales'].mean().mean():.2f}")
        print(f"  Promotion rate: {df['promo_flag'].mean():.1%}")
        print(f"  OOS rate: {(df['oos_loss_share'] > 0).mean():.1%}")

        return df


def generate_and_save_data(output_path: str = "data/sku_day_data.csv",
                          **kwargs) -> pd.DataFrame:
    """
    Generate synthetic data and save to CSV

    Args:
        output_path: Path to save CSV
        **kwargs: Arguments for SyntheticDataGenerator

    Returns:
        Generated DataFrame
    """
    generator = SyntheticDataGenerator(**kwargs)
    df = generator.generate_sku_day_data()

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nData saved to: {output_path}")
    print(f"File size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


if __name__ == "__main__":
    # Generate data
    df = generate_and_save_data(
        output_path="data/sku_day_data.csv",
        start_date="2022-01-01",
        n_days=730,
        categories=["Beverages", "Frozen_Foods", "Bakery"],
        skus_per_category=10,
        seed=42
    )

    # Display sample
    print("\nSample data (first 5 rows):")
    print(df.head())

    print("\nData summary:")
    print(df.describe())
