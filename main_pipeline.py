"""
Main Forecasting Pipeline

End-to-end pipeline for category-level daily forecasting MVP:
1. Generate/Load data
2. Aggregate SKU → Category
3. Engineer features
4. Fit ETS models
5. Train LightGBM on residuals
6. Generate forecasts
7. Evaluate performance
8. Create visualizations
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import (
    DATA_CONFIG, SPLIT_CONFIG, ETS_CONFIG, LGBM_CONFIG,
    TUNING_CONFIG, FORECAST_CONFIG, OUTPUT_CONFIG, METRIC_TARGETS
)

# Import custom modules
from src.data_generator import generate_and_save_data
from src.aggregation import aggregate_to_category, validate_aggregated_data
from src.feature_engineering import engineer_features
from src.ets_model import fit_ets_models
from src.lgbm_model import LGBMResidualModel, split_train_val_test
from src.forecast import HybridForecaster
from src.evaluation import evaluate_by_category, evaluate_forecast_components
from src.visualization import create_all_visualizations
from src.hyperparameter_tuning import LGBMHyperparameterTuner, save_tuned_params, load_tuned_params


class ForecastingPipeline:
    """Complete forecasting pipeline (configured via config.py)"""

    def __init__(self):
        """Initialize pipeline with config from config.py"""
        # Load configuration
        self.data_path = DATA_CONFIG['data_path']
        self.generate_new_data = DATA_CONFIG['generate_new_data']
        self.output_dir = OUTPUT_CONFIG['output_dir']
        self.tune_hyperparameters = TUNING_CONFIG['tune_hyperparameters']
        self.tuning_trials = TUNING_CONFIG['n_trials']

        # Create output directories
        os.makedirs(f"{OUTPUT_CONFIG['output_dir']}/forecasts", exist_ok=True)
        os.makedirs(f"{OUTPUT_CONFIG['output_dir']}/plots", exist_ok=True)
        os.makedirs(OUTPUT_CONFIG['model_dir'], exist_ok=True)
        os.makedirs(OUTPUT_CONFIG['data_dir'], exist_ok=True)

        # Pipeline components (will be populated)
        self.sku_df = None
        self.category_df = None
        self.feature_df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.ets_decomposer = None
        self.lgbm_model = None
        self.forecaster = None
        self.forecast_df = None
        self.metrics_df = None
        self.tuned_params = None

    def run_complete_pipeline(self) -> Dict:
        """
        Run complete forecasting pipeline (configured via config.py)

        Returns:
            Dictionary with pipeline results
        """
        # Load split configuration
        train_days = SPLIT_CONFIG['train_days']
        val_days = SPLIT_CONFIG['val_days']
        test_days = SPLIT_CONFIG['test_days']
        forecast_horizon = FORECAST_CONFIG['forecast_horizon']
        print("\n" + "="*80)
        print("CATEGORY-LEVEL DAILY FORECASTING PIPELINE (MVP)")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")

        # Step 1: Data Loading/Generation
        self.step_1_load_data()

        # Step 2: Aggregation
        self.step_2_aggregate_data()

        # Step 3: Feature Engineering
        self.step_3_engineer_features()

        # Step 4: Train/Val/Test Split
        self.step_4_split_data(train_days, val_days, test_days)

        # Step 5: ETS Decomposition
        self.step_5_fit_ets()

        # Step 5.5: Hyperparameter Tuning (optional)
        if self.tune_hyperparameters:
            self.step_5_5_tune_hyperparameters()
        else:
            # Try to load existing tuned parameters
            if os.path.exists(TUNING_CONFIG['tuned_params_path']):
                print("\n" + "="*80)
                print(f"Loading existing tuned parameters from {TUNING_CONFIG['tuned_params_path']}")
                print("="*80)
                self.tuned_params = load_tuned_params(TUNING_CONFIG['tuned_params_path'])
                print(f"Loaded tuned parameters: {list(self.tuned_params.keys())}")

        # Step 6: LightGBM Training
        self.step_6_train_lgbm()

        # Step 7: Generate Forecast
        self.step_7_generate_forecast(forecast_horizon)

        # Step 8: Evaluate
        self.step_8_evaluate()

        # Step 9: Visualize
        self.step_9_visualize()

        # Step 10: Save Results
        self.step_10_save_results()

        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nOutputs saved to: {self.output_dir}/")
        print("="*80 + "\n")

        return {
            'metrics': self.metrics_df,
            'forecast': self.forecast_df,
            'ets_decomposer': self.ets_decomposer,
            'lgbm_model': self.lgbm_model
        }

    def step_1_load_data(self):
        """Step 1: Load or generate data"""
        print("\n" + "="*80)
        print("STEP 1: DATA LOADING/GENERATION")
        print("="*80)

        if self.generate_new_data:
            self.sku_df, self.weather_df = generate_and_save_data(
                sku_output_path=f"{OUTPUT_CONFIG['data_dir']}/sku_day_data.csv",
                weather_output_path=f"{OUTPUT_CONFIG['data_dir']}/category_day_weather.csv",
                start_date=DATA_CONFIG['start_date'],
                n_days=DATA_CONFIG['n_days'],
                categories=DATA_CONFIG['categories'],
                skus_per_category=DATA_CONFIG['skus_per_category'],
                seed=DATA_CONFIG['seed']
            )
        else:
            print(f"\nLoading existing SKU data from: {self.data_path}")
            self.sku_df = pd.read_csv(self.data_path)
            self.sku_df['date'] = pd.to_datetime(self.sku_df['date'])
            print(f"Loaded {len(self.sku_df):,} SKU-day records")

            # Load weather data
            weather_path = f"{OUTPUT_CONFIG['data_dir']}/category_day_weather.csv"
            print(f"Loading weather data from: {weather_path}")
            self.weather_df = pd.read_csv(weather_path)
            self.weather_df['date'] = pd.to_datetime(self.weather_df['date'])
            print(f"Loaded {len(self.weather_df):,} category-day weather records")

        print("\n✓ Step 1 complete")

    def step_2_aggregate_data(self):
        """Step 2: Aggregate SKU → Category"""
        print("\n" + "="*80)
        print("STEP 2: DATA AGGREGATION (SKU → CATEGORY)")
        print("="*80)

        self.category_df = aggregate_to_category(self.sku_df, self.weather_df)
        validate_aggregated_data(self.category_df)

        # Save aggregated data
        self.category_df.to_csv(f"{OUTPUT_CONFIG['data_dir']}/category_day_data.csv", index=False)

        print("\n✓ Step 2 complete")

    def step_3_engineer_features(self):
        """Step 3: Engineer features"""
        print("\n" + "="*80)
        print("STEP 3: FEATURE ENGINEERING")
        print("="*80)

        self.feature_df = engineer_features(self.category_df)

        # Save features
        self.feature_df.to_csv(f"{OUTPUT_CONFIG['data_dir']}/category_day_features.csv", index=False)

        print("\n✓ Step 3 complete")

    def step_4_split_data(self, train_days, val_days, test_days):
        """Step 4: Create train/val/test splits"""
        print("\n" + "="*80)
        print("STEP 4: TRAIN/VAL/TEST SPLIT")
        print("="*80)

        self.train_df, self.val_df, self.test_df = split_train_val_test(
            self.feature_df, train_days, val_days, test_days
        )

        print("\nCreating train/val/test splits...")
        print(f"  Total unique dates: {len(self.feature_df['date'].unique())}")
        print(f"  Train: {train_days} days")
        print(f"  Val: {val_days} days")
        print(f"  Test: {test_days} days")

        print("\nSplit summary:")
        print(f"  Train: {len(self.train_df)} records ({self.train_df['date'].min()} to {self.train_df['date'].max()})")
        print(f"  Val: {len(self.val_df)} records ({self.val_df['date'].min()} to {self.val_df['date'].max()})")
        print(f"  Test: {len(self.test_df)} records ({self.test_df['date'].min()} to {self.test_df['date'].max()})")

        print("\n✓ Step 4 complete")

    def step_5_fit_ets(self):
        """Step 5: Fit ETS models"""
        print("\n" + "="*80)
        print("STEP 5: ETS DECOMPOSITION")
        print("="*80)

        # Fit on train + val data
        train_val_df = pd.concat([self.train_df, self.val_df], ignore_index=True)

        self.feature_df, self.ets_decomposer = fit_ets_models(
            train_val_df,
            seasonal_periods=ETS_CONFIG['seasonal_periods'],
            trend=ETS_CONFIG['trend'],
            seasonal=ETS_CONFIG['seasonal'],
            damped_trend=ETS_CONFIG['damped_trend'],
            smoothing_level=ETS_CONFIG['smoothing_level'],
            smoothing_trend=ETS_CONFIG['smoothing_trend'],
            smoothing_seasonal=ETS_CONFIG['smoothing_seasonal']
        )

        # Update splits with ETS components
        train_val_mask = self.feature_df['date'].isin(train_val_df['date'])
        test_mask = self.feature_df['date'].isin(self.test_df['date'])

        self.train_df = self.feature_df[
            self.feature_df['date'].isin(self.train_df['date'])
        ].copy()
        self.val_df = self.feature_df[
            self.feature_df['date'].isin(self.val_df['date'])
        ].copy()

        print("\n✓ Step 5 complete")

    def step_5_5_tune_hyperparameters(self):
        """Step 5.5: Tune hyperparameters (optional)"""
        print("\n" + "="*80)
        print("STEP 5.5: HYPERPARAMETER TUNING (OPTIONAL)")
        print("="*80)

        # Prepare feature columns
        exclude_cols = [
            'date', 'category', 'sales',
            'ets_fitted', 'ets_residual', 'actual', 'holiday_name'
        ]
        feature_cols = [col for col in self.feature_df.columns if col not in exclude_cols]

        # Use train + val data for tuning
        train_val_df = pd.concat([self.train_df, self.val_df], ignore_index=True)
        train_val_df = train_val_df.dropna(subset=['ets_residual'])

        # Initialize tuner
        tuner = LGBMHyperparameterTuner(
            n_trials=TUNING_CONFIG['n_trials'],
            cv_splits=TUNING_CONFIG['cv_splits'],
            min_train_size=TUNING_CONFIG['min_train_size'],
            val_size=TUNING_CONFIG['val_size']
        )

        # Tune pooled model
        best_params = tuner.tune(train_val_df, feature_cols, verbose=True)
        self.tuned_params = {'pooled': best_params}

        # Save tuned parameters
        save_tuned_params(self.tuned_params, TUNING_CONFIG['tuned_params_path'])

        print("\n✓ Step 5.5 complete")

    def step_6_train_lgbm(self):
        """Step 6: Train LightGBM on residuals"""
        print("\n" + "="*80)
        print("STEP 6: LIGHTGBM TRAINING")
        print("="*80)

        # Create masks for train/val
        train_val_df = pd.concat([self.train_df, self.val_df], ignore_index=True)
        train_mask = train_val_df['date'].isin(self.train_df['date']).values
        val_mask = train_val_df['date'].isin(self.val_df['date']).values

        # Train LightGBM (pooled model)
        if self.tuned_params:
            print("\nUsing tuned hyperparameters from Step 5.5")
            self.lgbm_model = LGBMResidualModel(tuned_params=self.tuned_params)
        else:
            print("\nUsing default hyperparameters from config")
            self.lgbm_model = LGBMResidualModel(params=LGBM_CONFIG['params'])

        self.lgbm_model.train(
            train_val_df,
            train_mask,
            val_mask,
            num_boost_round=LGBM_CONFIG['num_boost_round'],
            early_stopping_rounds=LGBM_CONFIG['early_stopping_rounds']
        )

        # Save model
        joblib.dump(self.lgbm_model, f"{OUTPUT_CONFIG['model_dir']}/lgbm_model.pkl")

        print("\n✓ Step 6 complete")

    def step_7_generate_forecast(self, forecast_horizon):
        """Step 7: Generate forecasts"""
        print("\n" + "="*80)
        print("STEP 7: FORECAST GENERATION")
        print("="*80)

        # Create hybrid forecaster
        self.forecaster = HybridForecaster(self.ets_decomposer, self.lgbm_model)

        # Use train + val as historical data for forecasting
        historical_df = pd.concat([self.train_df, self.val_df], ignore_index=True)

        # Generate forecasts (using test period dates)
        # For MVP, we'll use test data as "forecast period" to evaluate
        self.forecast_df = self.forecaster.forecast_all_categories(
            historical_df,
            forecast_horizon=len(self.test_df) // len(self.test_df['category'].unique())
        )

        # Save forecast
        self.forecast_df.to_csv(f'{self.output_dir}/forecasts/forecast_results.csv', index=False)

        print("\n✓ Step 7 complete")

    def step_8_evaluate(self):
        """Step 8: Evaluate forecasts"""
        print("\n" + "="*80)
        print("STEP 8: FORECAST EVALUATION")
        print("="*80)

        # Merge test actuals with forecasts
        eval_df = self.test_df.merge(
            self.forecast_df[['date', 'category', 'ets_forecast',
                             'lgbm_residual_forecast', 'final_forecast']],
            on=['date', 'category'],
            how='left'
        )

        # Evaluate by category
        self.metrics_df = evaluate_by_category(
            eval_df,
            actual_col='sales',
            predicted_col='final_forecast'
        )

        # Evaluate components
        component_metrics = evaluate_forecast_components(
            eval_df,
            actual_col='sales',
            ets_col='ets_forecast',
            final_col='final_forecast'
        )

        # Save metrics
        self.metrics_df.to_csv(f'{self.output_dir}/evaluation_metrics.csv', index=False)
        component_metrics.to_csv(f'{self.output_dir}/component_metrics.csv', index=False)

        print("\n✓ Step 8 complete")

    def step_9_visualize(self):
        """Step 9: Create visualizations"""
        print("\n" + "="*80)
        print("STEP 9: VISUALIZATION")
        print("="*80)

        # Merge test with forecast for plotting
        plot_df = self.test_df.merge(
            self.forecast_df[['date', 'category', 'ets_forecast',
                             'lgbm_residual_forecast', 'final_forecast']],
            on=['date', 'category'],
            how='left'
        )

        # Create all visualizations
        create_all_visualizations(
            self.train_df,
            self.test_df,
            self.forecast_df,
            self.metrics_df,
            output_dir=f'{self.output_dir}/plots'
        )

        print("\n✓ Step 9 complete")

    def step_10_save_results(self):
        """Step 10: Save all results and models"""
        print("\n" + "="*80)
        print("STEP 10: SAVING RESULTS")
        print("="*80)

        # Save ETS decomposer
        joblib.dump(self.ets_decomposer, f"{OUTPUT_CONFIG['model_dir']}/ets_decomposer.pkl")
        print("  ✓ ETS decomposer saved")

        # LGBM already saved in step 6
        print("  ✓ LightGBM model saved")

        # Save feature importance
        feature_importance = self.lgbm_model.feature_importance.head(20)
        feature_importance.to_csv(f'{self.output_dir}/feature_importance.csv', index=False)
        print("  ✓ Feature importance saved")

        # Create summary report
        self._create_summary_report()
        print("  ✓ Summary report created")

        print("\n✓ Step 10 complete")

    def _create_summary_report(self):
        """Create summary report"""
        report_path = f'{self.output_dir}/SUMMARY_REPORT.txt'

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CATEGORY-LEVEL DAILY FORECASTING - MVP SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("DATA SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total SKU-day records: {len(self.sku_df):,}\n")
            f.write(f"Total Category-day records: {len(self.category_df):,}\n")
            f.write(f"Categories: {', '.join(self.category_df['category'].unique())}\n")
            f.write(f"Date range: {self.category_df['date'].min().date()} to {self.category_df['date'].max().date()}\n\n")

            f.write("EVALUATION METRICS\n")
            f.write("-"*80 + "\n")
            f.write(self.metrics_df[['category', 'MAPE', 'MAE', 'MASE', 'Bias%', 'R2']].to_string(index=False))
            f.write("\n\n")

            avg_mape = self.metrics_df['MAPE'].mean()
            avg_mase = self.metrics_df['MASE'].mean()

            f.write("MVP SUCCESS CRITERIA\n")
            f.write("-"*80 + "\n")
            mape_target = METRIC_TARGETS['mape_target']
            mase_target = METRIC_TARGETS['mase_target']
            f.write(f"Average MAPE: {avg_mape:.2f}% (Target: <{mape_target}%) {'✓' if avg_mape < mape_target else '✗'}\n")
            f.write(f"Average MASE: {avg_mase:.3f} (Target: <{mase_target}) {'✓' if avg_mase < mase_target else '✗'}\n\n")

            if avg_mape < mape_target and avg_mase < mase_target:
                f.write("STATUS: ✓ MVP SUCCESS CRITERIA MET!\n")
            else:
                f.write("STATUS: ⚠ MVP success criteria not fully met\n")

            f.write("\n" + "="*80 + "\n")

        print(f"\n  Summary report saved to: {report_path}")


def main():
    """Main entry point - all configuration is in config.py"""
    # Initialize and run pipeline (uses config.py)
    pipeline = ForecastingPipeline()

    # Run complete pipeline (uses config.py)
    results = pipeline.run_complete_pipeline()

    print("\n" + "="*80)
    print("SUCCESS! Complete forecasting pipeline executed.")
    print("="*80)
    print("\nKey Outputs:")
    print("  - Forecasts: outputs/forecasts/")
    print("  - Plots: outputs/plots/")
    print("  - Models: models/")
    print("  - Metrics: outputs/evaluation_metrics.csv")
    print("  - Summary: outputs/SUMMARY_REPORT.txt")
    print("="*80 + "\n")

    return results


if __name__ == "__main__":
    main()
