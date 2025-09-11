"""
Example usage of Time Series Forecasting
"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import swat
from models.time_series_forecasting import TimeSeriesForecasting

def sales_forecasting_pipeline():
    """Complete example with sales forecasting"""
    
    # Connect to CAS
    cas = swat.CAS('your-cas-server.com', 5570)
    
    try:
        # Initialize forecasting
        ts_forecaster = TimeSeriesForecasting(cas)
        
        # Sample sales data
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
        sales_data = pd.DataFrame({
            'date': dates,
            'sales': np.random.normal(1000, 200, len(dates)) + 
                    np.sin(np.arange(len(dates)) * 2 * np.pi / 12) * 100,  # Seasonal pattern
            'product_category': ['Electronics'] * len(dates)
        })
        
        # Prepare data
        ts_table = ts_forecaster.prepare_time_series_data(
            sales_data, 'date', 'sales', 'product_category'
        )
        
        # Generate forecasts
        forecast_result, forecast_df = ts_forecaster.generate_forecasts(ts_table, 12)
        
        # Ensemble forecasting
        models_config = {
            'ARIMAX': {
                'id': 'date_formatted',
                'target': 'sales',
                'forecast': 12,
                'p': 2, 'd': 1, 'q': 2
            },
            'ESM': {
                'id': 'date_formatted',
                'target': 'sales',
                'forecast': 12,
                'trend': 'LINEAR',
                'season': 'ADDITIVE'
            },
            'UCM': {
                'id': 'date_formatted',
                'target': 'sales',
                'forecast': 12,
                'level': True,
                'season': {'length': 12}
            }
        }
        
        ensemble_result = ts_forecaster.ensemble_forecasting(ts_table, models_config)
        ensemble_df = ensemble_result.to_frame()
        
        print("Time series forecasting completed successfully")
        print(f"Generated {len(ensemble_df)} forecast periods")
        
        return forecast_df, ensemble_df
        
    finally:
        cas.close()

if __name__ == "__main__":
    forecast_df, ensemble_df = sales_forecasting_pipeline()