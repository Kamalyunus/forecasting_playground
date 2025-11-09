"""
Synthetic Data Generator

Generates SKU-day and Category-day level data matching the production format:
- SKU-day: date, sku_id, category, sales, instock_rate, high_impact_promo, final_sku_price
- Category-day: date, category, avg_temperature, avg_rainfall, avg_snowfall, holiday_flag
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DATA_CONFIG


class SyntheticDataGenerator:
    """Generate synthetic SKU-day and Category-day level data"""

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
            n_days: Number of days to generate
            categories: List of category names
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

        # Define holidays
        self.holidays = self._define_holidays()

    def _define_holidays(self) -> List[pd.Timestamp]:
        """Define major holidays across multiple years"""
        holidays = []
        years = list(range(2020, 2026))  # Cover wide range

        for year in years:
            holidays.extend([
                pd.Timestamp(f'{year}-01-01'),  # New Year
                pd.Timestamp(f'{year}-07-04'),  # July 4th
                pd.Timestamp(f'{year}-12-25'),  # Christmas
            ])

        # Thanksgiving (4th Thursday of November)
        for year in years:
            nov_first = pd.Timestamp(f'{year}-11-01')
            first_thursday = nov_first + pd.Timedelta(days=(3 - nov_first.weekday()) % 7)
            thanksgiving = first_thursday + pd.Timedelta(weeks=3)
            holidays.append(thanksgiving)

        return holidays

    def _get_holiday_flag(self, date: pd.Timestamp) -> int:
        """
        Get holiday flag for a given date

        Returns 14 to -5 for holiday period, -100 for regular days

        Args:
            date: Date to check

        Returns:
            Holiday flag value
        """
        min_distance = 999

        for holiday_date in self.holidays:
            distance = (holiday_date - date).days
            if abs(distance) < abs(min_distance):
                min_distance = distance

        # Within holiday window (14 days before to 5 days after)
        if -5 <= min_distance <= 14:
            return min_distance
        else:
            return -100

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
        # Weekly pattern (weekend higher)
        day_of_week = date.dayofweek
        weekday_pattern = {0: 0.9, 1: 0.85, 2: 0.88, 3: 0.92, 4: 1.1, 5: 1.3, 6: 1.25}
        dow_factor = weekday_pattern.get(day_of_week, 1.0)

        # Annual seasonality (category-specific)
        day_of_year = date.dayofyear

        if category == "Beverages":
            # Peak in summer
            annual_factor = 1.0 + 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
        elif category == "Frozen_Foods":
            # Higher in summer and winter
            annual_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
        else:  # Bakery
            # More stable
            annual_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (day_of_year - 100) / 365.25)

        base_demand = sku_base_level * dow_factor * annual_factor
        return base_demand

    def generate_sku_day_data(self) -> pd.DataFrame:
        """
        Generate SKU-day level data

        Returns:
            DataFrame with columns: date, sku_id, category, sales,
                                   instock_rate, high_impact_promo, final_sku_price
        """
        print("\nGenerating synthetic SKU-day level data...")
        print(f"  Categories: {len(self.categories)}")
        print(f"  SKUs per category: {self.skus_per_category}")
        print(f"  Date range: {self.start_date.date()} to {self.dates[-1].date()}")
        print(f"  Total days: {self.n_days}")

        records = []

        # Generate SKU metadata
        sku_info = {}
        for category in self.categories:
            # Category-specific price ranges
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

        # Generate data for each date and SKU
        for date in self.dates:
            # Get holiday flag for this date
            holiday_flag = self._get_holiday_flag(date)
            is_holiday_period = holiday_flag != -100

            for sku_id, info in sku_info.items():
                category = info['category']
                base_price = info['base_price']
                base_volume = info['base_volume']

                # Base demand
                base_demand = self._generate_base_demand(date, category, base_volume)

                # High impact promo (more likely on weekends and holidays)
                promo_prob = 0.05
                if date.dayofweek >= 5:
                    promo_prob += 0.08
                if is_holiday_period:
                    promo_prob += 0.12

                high_impact_promo = 1 if np.random.random() < promo_prob else 0

                # Promo lift
                if high_impact_promo:
                    promo_lift = np.random.uniform(1.8, 3.0)
                    discount_pct = np.random.uniform(0.15, 0.40)
                else:
                    promo_lift = 1.0
                    discount_pct = 0.0

                # Holiday lift
                if is_holiday_period:
                    if 1 <= holiday_flag <= 14:  # Days before holiday
                        holiday_lift = 1.2 + (14 - holiday_flag) / 28  # Ramps up
                    elif holiday_flag == 0:  # Holiday day
                        holiday_lift = 1.5
                    else:  # Days after holiday (-1 to -5)
                        holiday_lift = 0.85  # Dip after holiday
                else:
                    holiday_lift = 1.0

                # Calculate true demand
                true_demand = base_demand * promo_lift * holiday_lift

                # Add noise
                noise = np.random.normal(1.0, 0.15)
                true_demand *= noise
                true_demand = max(0, true_demand)

                # Instock rate (0-100%)
                # Most days have high instock, occasional low stock
                if np.random.random() < 0.08:  # 8% chance of stockout issue
                    instock_rate = np.random.uniform(30, 80)  # Partial stockout
                else:
                    instock_rate = np.random.uniform(85, 100)  # Good stock

                # Sales affected by instock
                sales = true_demand * (instock_rate / 100.0)

                # Final SKU price
                final_sku_price = base_price * (1 - discount_pct)

                # Create record
                record = {
                    'date': date,
                    'sku_id': sku_id,
                    'category': category,
                    'sales': round(sales, 2),
                    'instock_rate': round(instock_rate, 2),
                    'high_impact_promo': high_impact_promo,
                    'final_sku_price': round(final_sku_price, 2)
                }

                records.append(record)

        df = pd.DataFrame(records)

        print(f"\n  Generated {len(df):,} SKU-day records")
        print(f"  Avg daily sales per SKU: {df.groupby('sku_id')['sales'].mean().mean():.2f}")
        print(f"  High-impact promo rate: {df['high_impact_promo'].mean():.1%}")
        print(f"  Avg instock rate: {df['instock_rate'].mean():.1f}%")
        print(f"  Low instock rate (<60%): {(df['instock_rate'] < 60).mean():.1%}")

        return df

    def generate_category_day_weather(self) -> pd.DataFrame:
        """
        Generate category-day level weather and holiday data

        Returns:
            DataFrame with columns: date, category, avg_temperature,
                                   avg_rainfall, avg_snowfall, holiday_flag
        """
        print("\nGenerating category-day weather/holiday data...")

        records = []

        for date in self.dates:
            # Generate weather once per day (same for all categories)
            day_of_year = date.dayofyear

            # Temperature (seasonal pattern)
            seasonal_temp = 60 + 30 * np.sin(2 * np.pi * (day_of_year - 100) / 365.25)
            daily_variation = np.random.normal(0, 10)
            avg_temperature = np.clip(seasonal_temp + daily_variation, 20, 100)

            # Rainfall (more in spring/fall)
            seasonal_rain_prob = 0.2 + 0.15 * np.sin(2 * np.pi * (day_of_year - 50) / 365.25)
            if np.random.random() < seasonal_rain_prob:
                avg_rainfall = np.random.gamma(2, 0.3)
            else:
                avg_rainfall = 0.0

            # Snowfall (only in winter)
            if seasonal_temp < 35:
                snow_prob = 0.15
                if np.random.random() < snow_prob:
                    avg_snowfall = np.random.gamma(1.5, 0.5)
                else:
                    avg_snowfall = 0.0
            else:
                avg_snowfall = 0.0

            # Get holiday flag
            holiday_flag = self._get_holiday_flag(date)

            # Create record for each category
            for category in self.categories:
                record = {
                    'date': date,
                    'category': category,
                    'avg_temperature': round(avg_temperature, 1),
                    'avg_rainfall': round(avg_rainfall, 2),
                    'avg_snowfall': round(avg_snowfall, 2),
                    'holiday_flag': holiday_flag
                }
                records.append(record)

        df = pd.DataFrame(records)

        print(f"  Generated {len(df):,} category-day records")
        print(f"  Avg temperature: {df['avg_temperature'].mean():.1f}°F")
        print(f"  Rainy days: {(df['avg_rainfall'] > 0).mean():.1%}")
        print(f"  Snowy days: {(df['avg_snowfall'] > 0).mean():.1%}")
        print(f"  Holiday period days: {(df['holiday_flag'] != -100).mean():.1%}")

        return df


def generate_and_save_data(sku_output_path: str = "data/sku_day_data.csv",
                          weather_output_path: str = "data/category_day_weather.csv",
                          **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic data and save to CSV files

    Args:
        sku_output_path: Path to save SKU-day data
        weather_output_path: Path to save category-day weather data
        **kwargs: Arguments for SyntheticDataGenerator

    Returns:
        Tuple of (sku_df, weather_df)
    """
    print("="*60)
    print("GENERATING SYNTHETIC DATA")
    print("="*60)

    generator = SyntheticDataGenerator(**kwargs)

    # Generate SKU-day data
    sku_df = generator.generate_sku_day_data()

    # Generate category-day weather data
    weather_df = generator.generate_category_day_weather()

    # Save to CSV
    import os
    os.makedirs(os.path.dirname(sku_output_path) if os.path.dirname(sku_output_path) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(weather_output_path) if os.path.dirname(weather_output_path) else '.', exist_ok=True)

    sku_df.to_csv(sku_output_path, index=False)
    weather_df.to_csv(weather_output_path, index=False)

    print(f"\n  SKU-day data saved to: {sku_output_path}")
    print(f"  Weather data saved to: {weather_output_path}")
    print(f"  Total size: {(sku_df.memory_usage(deep=True).sum() + weather_df.memory_usage(deep=True).sum()) / 1024**2:.2f} MB")

    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE")
    print("="*60)

    return sku_df, weather_df
