"""
Visualization Module

Comprehensive visualizations for forecasting analysis:
- Time series plots (actual vs forecast)
- Residual analysis
- Component decomposition
- Error distribution
- Feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10


def plot_forecast_vs_actual(df: pd.DataFrame,
                            category: str,
                            train_df: Optional[pd.DataFrame] = None,
                            actual_col: str = 'sales',
                            forecast_col: str = 'final_forecast',
                            save_path: Optional[str] = None) -> None:
    """
    Plot actual vs forecast time series

    Args:
        df: Test/forecast DataFrame
        category: Category name
        train_df: Optional training data to show context
        actual_col: Actual values column
        forecast_col: Forecast values column
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(15, 6))

    # Filter category
    cat_test = df[df['category'] == category].sort_values('date')

    # Plot training data if provided
    if train_df is not None:
        cat_train = train_df[train_df['category'] == category].sort_values('date')
        # Show last 90 days of training
        cat_train_recent = cat_train.tail(90)
        ax.plot(cat_train_recent['date'], cat_train_recent[actual_col],
               label='Historical', alpha=0.5, color='gray', linewidth=1)

    # Plot test actual and forecast
    ax.plot(cat_test['date'], cat_test[actual_col],
           label='Actual', linewidth=2, marker='o', markersize=4, alpha=0.8)

    ax.plot(cat_test['date'], cat_test[forecast_col],
           label='Forecast', linewidth=2, marker='s', markersize=4,
           linestyle='--', alpha=0.8)

    # Add vertical line to separate train/test
    if train_df is not None:
        ax.axvline(x=cat_test['date'].iloc[0], color='red',
                  linestyle=':', alpha=0.5, label='Forecast Start')

    ax.set_title(f'{category}: Actual vs Forecast', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sales', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.close()


def plot_forecast_components(df: pd.DataFrame,
                            category: str,
                            actual_col: str = 'sales',
                            ets_col: str = 'ets_forecast',
                            lgbm_col: str = 'lgbm_residual_forecast',
                            final_col: str = 'final_forecast',
                            save_path: Optional[str] = None) -> None:
    """
    Plot forecast components (ETS, LGBM residual, final)

    Args:
        df: Forecast DataFrame
        category: Category name
        actual_col: Actual values column
        ets_col: ETS forecast column
        lgbm_col: LGBM residual column
        final_col: Final forecast column
        save_path: Path to save plot
    """
    cat_df = df[df['category'] == category].sort_values('date')

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # Plot 1: ETS only vs Actual
    axes[0].plot(cat_df['date'], cat_df[actual_col],
                label='Actual', linewidth=2, marker='o', markersize=4, alpha=0.8)
    axes[0].plot(cat_df['date'], cat_df[ets_col],
                label='ETS Forecast', linewidth=2, linestyle='--', alpha=0.8)
    axes[0].set_title(f'{category}: ETS Component', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Sales', fontsize=10)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: LGBM Residual
    axes[1].plot(cat_df['date'], cat_df[lgbm_col],
                label='LGBM Residual', linewidth=2, color='green', alpha=0.7)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title(f'{category}: LGBM Residual Component', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Residual', fontsize=10)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Final combined forecast vs Actual
    axes[2].plot(cat_df['date'], cat_df[actual_col],
                label='Actual', linewidth=2, marker='o', markersize=4, alpha=0.8)
    axes[2].plot(cat_df['date'], cat_df[final_col],
                label='Final Forecast (ETS + LGBM)', linewidth=2,
                linestyle='--', marker='s', markersize=4, alpha=0.8)
    axes[2].set_title(f'{category}: Final Combined Forecast', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Date', fontsize=10)
    axes[2].set_ylabel('Sales', fontsize=10)
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.close()


def plot_residual_analysis(df: pd.DataFrame,
                          category: str,
                          actual_col: str = 'sales',
                          forecast_col: str = 'final_forecast',
                          save_path: Optional[str] = None) -> None:
    """
    Plot residual analysis (residuals over time, distribution, Q-Q plot)

    Args:
        df: DataFrame with actual and forecast
        category: Category name
        actual_col: Actual values column
        forecast_col: Forecast values column
        save_path: Path to save plot
    """
    cat_df = df[df['category'] == category].sort_values('date')

    # Calculate residuals
    residuals = cat_df[actual_col].values - cat_df[forecast_col].values

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Residuals over time
    axes[0, 0].plot(cat_df['date'], residuals, marker='o', linestyle='-', alpha=0.7)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Residuals Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, alpha=0.3)
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)

    # Plot 2: Residual distribution
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Residual Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    # Add statistics
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    axes[0, 1].text(0.05, 0.95, f'Mean: {mean_res:.2f}\nStd: {std_res:.2f}',
                   transform=axes[0, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Actual vs Predicted scatter
    axes[1, 0].scatter(cat_df[actual_col], cat_df[forecast_col], alpha=0.6)

    # Perfect prediction line
    min_val = min(cat_df[actual_col].min(), cat_df[forecast_col].min())
    max_val = max(cat_df[actual_col].max(), cat_df[forecast_col].max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val],
                   'r--', alpha=0.5, label='Perfect Prediction')

    axes[1, 0].set_title('Actual vs Predicted', fontweight='bold')
    axes[1, 0].set_xlabel('Actual')
    axes[1, 0].set_ylabel('Predicted')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Absolute percentage error over time
    ape = np.abs(residuals / cat_df[actual_col].values) * 100
    ape = ape[np.isfinite(ape)]  # Remove inf/nan

    axes[1, 1].plot(cat_df['date'].values[:len(ape)], ape,
                   marker='o', linestyle='-', alpha=0.7, color='orange')
    axes[1, 1].axhline(y=np.mean(ape), color='red', linestyle='--',
                      alpha=0.5, label=f'Mean APE: {np.mean(ape):.1f}%')
    axes[1, 1].set_title('Absolute Percentage Error Over Time', fontweight='bold')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('APE (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

    fig.suptitle(f'{category}: Residual Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.close()


def plot_metrics_comparison(metrics_df: pd.DataFrame,
                           save_path: Optional[str] = None) -> None:
    """
    Plot comparison of metrics across categories

    Args:
        metrics_df: DataFrame with metrics by category
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # MAPE comparison
    axes[0, 0].bar(metrics_df['category'], metrics_df['MAPE'], alpha=0.7, color='steelblue')
    axes[0, 0].axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Target: 20%')
    axes[0, 0].set_title('MAPE by Category', fontweight='bold')
    axes[0, 0].set_ylabel('MAPE (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)

    # MAE comparison
    axes[0, 1].bar(metrics_df['category'], metrics_df['MAE'], alpha=0.7, color='coral')
    axes[0, 1].set_title('MAE by Category', fontweight='bold')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)

    # MASE comparison
    axes[1, 0].bar(metrics_df['category'], metrics_df['MASE'], alpha=0.7, color='seagreen')
    axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Target: 1.0')
    axes[1, 0].set_title('MASE by Category', fontweight='bold')
    axes[1, 0].set_ylabel('MASE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

    # Bias comparison
    axes[1, 1].bar(metrics_df['category'], metrics_df['Bias%'], alpha=0.7, color='purple')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Bias by Category', fontweight='bold')
    axes[1, 1].set_ylabel('Bias (%)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

    fig.suptitle('Forecast Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    plt.close()


def create_all_visualizations(train_df: pd.DataFrame,
                             test_df: pd.DataFrame,
                             forecast_df: pd.DataFrame,
                             metrics_df: pd.DataFrame,
                             output_dir: str = 'outputs/plots') -> None:
    """
    Create all visualizations for MVP

    Args:
        train_df: Training data
        test_df: Test data (with actuals)
        forecast_df: Forecast data (with predictions)
        metrics_df: Evaluation metrics by category
        output_dir: Directory to save plots
    """
    print("="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # Merge test and forecast for plotting
    plot_df = test_df.merge(
        forecast_df[['date', 'category', 'ets_forecast', 'lgbm_residual_forecast', 'final_forecast']],
        on=['date', 'category'],
        how='left'
    )

    categories = plot_df['category'].unique()

    for category in categories:
        print(f"\n  Generating plots for {category}...")

        # 1. Forecast vs Actual
        plot_forecast_vs_actual(
            plot_df, category, train_df,
            save_path=f'{output_dir}/forecast_vs_actual_{category}.png'
        )

        # 2. Forecast Components
        if 'ets_forecast' in plot_df.columns:
            plot_forecast_components(
                plot_df, category,
                save_path=f'{output_dir}/forecast_components_{category}.png'
            )

        # 3. Residual Analysis
        plot_residual_analysis(
            plot_df, category,
            save_path=f'{output_dir}/residual_analysis_{category}.png'
        )

        print(f"    ✓ Plots generated for {category}")

    # 4. Metrics Comparison across categories
    print("\n  Generating metrics comparison plot...")
    plot_metrics_comparison(
        metrics_df,
        save_path=f'{output_dir}/metrics_comparison.png'
    )

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"All plots saved to: {output_dir}/")
