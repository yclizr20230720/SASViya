import pandas as pd
import numpy as np
from datetime import datetime
import swat

class TimeSeriesForecasting:
    def __init__(self, cas_session):
        self.cas = cas_session
        self.forecast_results = {}
        
    def prepare_time_series_data(self, data, date_col, value_col, group_col=None):
        """Prepare data for time series analysis"""
        # Upload data to CAS
        cas_table = self.cas.upload(data, casout='ts_data_raw')
        
        # Prepare time series format
        self.cas.datastep.runcode(f"""
        data work.ts_prepared;
            set {cas_table.name};
            
            /* Ensure proper date format */
            date_formatted = input({date_col}, yymmdd10.);
            format date_formatted yymmdd10.;
            
            /* Create time series components */
            year = year(date_formatted);
            month = month(date_formatted);
            quarter = qtr(date_formatted);
            day_of_week = weekday(date_formatted);
            
            /* Handle missing values */
            if missing({value_col}) then {value_col} = 0;
            
            keep {'group_col + " " if group_col else ""}date_formatted {value_col} year month quarter day_of_week;
        run;
        """)
        
        return self.cas.CASTable('ts_prepared', caslib='work')
    
    def generate_forecasts(self, ts_table, forecast_periods=12):
        """Generate forecasts using SAS Viya time series procedures"""
        # Load time series action set
        self.cas.loadactionset('tsmodel')
        
        # Automatic model selection and forecasting
        forecast_result = self.cas.tsmodel.tsm(
            table=ts_table,
            by=['group_col'] if hasattr(ts_table, 'group_col') else None,
            id='date_formatted',
            target='value_col',
            seasonality=12,  # Monthly seasonality
            forecast=forecast_periods,
            holdout=6,  # Hold out last 6 periods for validation
            models=[
                'ARIMAX',
                'ESM',  # Exponential Smoothing
                'UCM',  # Unobserved Components Model
                'LSTM'  # Long Short-Term Memory
            ],
            casout={'name': 'forecast_results', 'replace': True}
        )
        
        # Get forecast results
        forecast_table = self.cas.CASTable('forecast_results')
        forecast_df = forecast_table.to_frame()
        
        return forecast_result, forecast_df
    
    def ensemble_forecasting(self, ts_table, models_config):
        """Create ensemble of multiple forecasting models"""
        ensemble_results = {}
        
        for model_name, config in models_config.items():
            if model_name == 'ARIMAX':
                result = self.cas.tsmodel.arimax(
                    table=ts_table,
                    **config,
                    casout={'name': f'forecast_{model_name.lower()}', 'replace': True}
                )
            elif model_name == 'ESM':
                result = self.cas.tsmodel.esm(
                    table=ts_table,
                    **config,
                    casout={'name': f'forecast_{model_name.lower()}', 'replace': True}
                )
            elif model_name == 'UCM':
                result = self.cas.tsmodel.ucm(
                    table=ts_table,
                    **config,
                    casout={'name': f'forecast_{model_name.lower()}', 'replace': True}
                )
            
            ensemble_results[model_name] = result
        
        # Combine forecasts using weighted average
        self.cas.datastep.runcode("""
        data work.ensemble_forecast;
            merge work.forecast_arimax(rename=(forecast=forecast_arimax))
                  work.forecast_esm(rename=(forecast=forecast_esm))
                  work.forecast_ucm(rename=(forecast=forecast_ucm));
            by date_formatted;
            
            /* Weighted ensemble (weights based on historical accuracy) */
            ensemble_forecast = 0.4 * forecast_arimax + 
                              0.35 * forecast_esm + 
                              0.25 * forecast_ucm;
            
            /* Calculate prediction intervals */
            ensemble_lower = ensemble_forecast * 0.9;
            ensemble_upper = ensemble_forecast * 1.1;
        run;
        """)
        
        return self.cas.CASTable('ensemble_forecast', caslib='work')