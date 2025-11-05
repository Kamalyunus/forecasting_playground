"""
Evaluation Module

Comprehensive evaluation metrics for forecasting:
- MAPE (Mean Absolute Percentage Error)
- MAE (Mean Absolute Error)
- MASE (Mean Absolute Scaled Error)
- Bias
- RMSE
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def calculate_mape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate Mean Absolute Percentage Error

    Args:
        actual: Actual values
        predicted: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE percentage
    """
    # Remove zero actuals to avoid division by zero
    mask = actual > epsilon
    if mask.sum() == 0:
        return np.nan

    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]

    mape = np.mean(np.abs((actual_filtered - predicted_filtered) / actual_filtered)) * 100

    return mape


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        MAE
    """
    mae = np.mean(np.abs(actual - predicted))
    return mae


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        RMSE
    """
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    return rmse


def calculate_mase(actual: np.ndarray,
                  predicted: np.ndarray,
                  seasonal_period: int = 7) -> float:
    """
    Calculate Mean Absolute Scaled Error

    MASE < 1: Better than naive seasonal forecast
    MASE = 1: Same as naive seasonal forecast
    MASE > 1: Worse than naive seasonal forecast

    Args:
        actual: Actual values
        predicted: Predicted values
        seasonal_period: Seasonal period for naive forecast

    Returns:
        MASE
    """
    # Forecast error
    forecast_error = np.abs(actual - predicted)
    mae_forecast = np.mean(forecast_error)

    # Naive seasonal forecast error (using in-sample data)
    if len(actual) <= seasonal_period:
        # Not enough data for seasonal naive
        return np.nan

    naive_errors = np.abs(actual[seasonal_period:] - actual[:-seasonal_period])
    mae_naive = np.mean(naive_errors)

    if mae_naive == 0:
        return np.nan

    mase = mae_forecast / mae_naive

    return mase


def calculate_bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate bias (tendency to over/under forecast)

    Positive: Over-forecasting
    Negative: Under-forecasting

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        Bias percentage
    """
    total_actual = np.sum(actual)
    if total_actual == 0:
        return np.nan

    bias = (np.sum(predicted) - total_actual) / total_actual * 100

    return bias


def evaluate_forecast(actual: np.ndarray,
                     predicted: np.ndarray,
                     category_name: Optional[str] = None,
                     seasonal_period: int = 7) -> Dict[str, float]:
    """
    Comprehensive forecast evaluation

    Args:
        actual: Actual values
        predicted: Predicted values
        category_name: Category name for display
        seasonal_period: Seasonal period for MASE

    Returns:
        Dictionary of metrics
    """
    # Calculate all metrics
    mape = calculate_mape(actual, predicted)
    mae = calculate_mae(actual, predicted)
    rmse = calculate_rmse(actual, predicted)
    mase = calculate_mase(actual, predicted, seasonal_period)
    bias = calculate_bias(actual, predicted)

    # R-squared
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    metrics = {
        'category': category_name,
        'n_samples': len(actual),
        'MAPE': round(mape, 2),
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MASE': round(mase, 3),
        'Bias%': round(bias, 2),
        'R2': round(r2, 3),
        'mean_actual': round(np.mean(actual), 2),
        'mean_predicted': round(np.mean(predicted), 2),
    }

    # Print results
    if category_name:
        print(f"\n{'='*60}")
        print(f"Category: {category_name}")
        print(f"{'='*60}")

    print(f"  Samples: {metrics['n_samples']}")
    print(f"  MAPE:    {metrics['MAPE']:.2f}% {'✓' if mape < 20 else '✗'}")
    print(f"  MAE:     {metrics['MAE']:.2f}")
    print(f"  RMSE:    {metrics['RMSE']:.2f}")
    print(f"  MASE:    {metrics['MASE']:.3f} {'✓' if mase < 1.0 else '✗'}")
    print(f"  Bias:    {metrics['Bias%']:.2f}%")
    print(f"  R²:      {metrics['R2']:.3f}")
    print(f"  Mean Actual:    {metrics['mean_actual']:.2f}")
    print(f"  Mean Predicted: {metrics['mean_predicted']:.2f}")

    if category_name:
        print(f"{'='*60}\n")

    return metrics


def evaluate_by_category(df: pd.DataFrame,
                        actual_col: str = 'sales',
                        predicted_col: str = 'final_forecast',
                        seasonal_period: int = 7) -> pd.DataFrame:
    """
    Evaluate forecasts by category

    Args:
        df: DataFrame with actual and predicted values
        actual_col: Column name for actual values
        predicted_col: Column name for predicted values
        seasonal_period: Seasonal period for MASE

    Returns:
        DataFrame with metrics by category
    """
    print("="*60)
    print("EVALUATION BY CATEGORY")
    print("="*60)

    categories = df['category'].unique()
    results = []

    for category in categories:
        cat_df = df[df['category'] == category]

        actual = cat_df[actual_col].values
        predicted = cat_df[predicted_col].values

        metrics = evaluate_forecast(actual, predicted, category, seasonal_period)
        results.append(metrics)

    # Create summary DataFrame
    results_df = pd.DataFrame(results)

    # Overall metrics
    print("\n" + "="*60)
    print("OVERALL METRICS")
    print("="*60)

    overall_actual = df[actual_col].values
    overall_predicted = df[predicted_col].values
    overall_metrics = evaluate_forecast(overall_actual, overall_predicted, "OVERALL", seasonal_period)

    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(results_df[['category', 'MAPE', 'MAE', 'MASE', 'Bias%', 'R2']].to_string(index=False))

    # MVP success criteria check
    print("\n" + "="*60)
    print("MVP SUCCESS CRITERIA")
    print("="*60)
    avg_mape = results_df['MAPE'].mean()
    avg_mase = results_df['MASE'].mean()

    print(f"  Avg MAPE: {avg_mape:.2f}% (Target: <20%) {'✓' if avg_mape < 20 else '✗'}")
    print(f"  Avg MASE: {avg_mase:.3f} (Target: <1.0) {'✓' if avg_mase < 1.0 else '✗'}")

    if avg_mape < 20 and avg_mase < 1.0:
        print("\n  🎉 MVP SUCCESS CRITERIA MET!")
    else:
        print("\n  ⚠ MVP success criteria not fully met - consider model refinement")

    return results_df


def evaluate_forecast_components(df: pd.DataFrame,
                                 actual_col: str = 'sales',
                                 ets_col: str = 'ets_forecast',
                                 lgbm_col: str = 'lgbm_residual_forecast',
                                 final_col: str = 'final_forecast') -> pd.DataFrame:
    """
    Evaluate individual components of hybrid forecast

    Args:
        df: DataFrame with actual and component forecasts
        actual_col: Actual values column
        ets_col: ETS forecast column
        lgbm_col: LGBM residual forecast column
        final_col: Final combined forecast column

    Returns:
        Component evaluation DataFrame
    """
    print("="*60)
    print("COMPONENT EVALUATION")
    print("="*60)

    categories = df['category'].unique()
    results = []

    for category in categories:
        cat_df = df[df['category'] == category]

        actual = cat_df[actual_col].values
        ets_only = cat_df[ets_col].values
        final = cat_df[final_col].values

        # Evaluate ETS only
        ets_mae = calculate_mae(actual, ets_only)
        ets_mape = calculate_mape(actual, ets_only)

        # Evaluate final (ETS + LGBM)
        final_mae = calculate_mae(actual, final)
        final_mape = calculate_mape(actual, final)

        # Improvement from adding LGBM
        mae_improvement = (ets_mae - final_mae) / ets_mae * 100 if ets_mae > 0 else 0
        mape_improvement = (ets_mape - final_mape) / ets_mape * 100 if ets_mape > 0 else 0

        results.append({
            'category': category,
            'ETS_MAE': round(ets_mae, 2),
            'ETS_MAPE': round(ets_mape, 2),
            'Final_MAE': round(final_mae, 2),
            'Final_MAPE': round(final_mape, 2),
            'MAE_Improvement%': round(mae_improvement, 2),
            'MAPE_Improvement%': round(mape_improvement, 2)
        })

    results_df = pd.DataFrame(results)

    print("\nComponent Performance:")
    print(results_df.to_string(index=False))

    avg_mae_improvement = results_df['MAE_Improvement%'].mean()
    avg_mape_improvement = results_df['MAPE_Improvement%'].mean()

    print(f"\nAverage improvements by adding LGBM:")
    print(f"  MAE:  {avg_mae_improvement:.2f}% improvement")
    print(f"  MAPE: {avg_mape_improvement:.2f}% improvement")

    return results_df


if __name__ == "__main__":
    # Test evaluation module
    print("Testing evaluation module...")

    # Create sample data
    np.random.seed(42)
    n = 100

    actual = np.random.uniform(100, 200, n)
    predicted = actual + np.random.normal(0, 10, n)

    # Evaluate
    metrics = evaluate_forecast(actual, predicted, "Test Category")

    print("\nTest completed successfully!")
