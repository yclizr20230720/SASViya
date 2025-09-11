"""
Example usage of Customer Churn Analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from connections.viya_connection import ViyaConnection
from models.churn_analysis import CustomerChurnAnalysis

def run_churn_analysis():
    """Example: Complete churn analysis pipeline"""
    
    # Initialize connection
    viya_conn = ViyaConnection()
    viya_conn.connect_swat('your-cas-server.com')
    viya_conn.connect_sasctl('your-viya-server.com', 'username', 'password')
    
    try:
        # Run analysis
        churn_analysis = CustomerChurnAnalysis(viya_conn)
        
        # Load and process data
        df = churn_analysis.load_data_from_viya('customer_data')
        df_processed = churn_analysis.feature_engineering(df)
        
        # Build and evaluate model
        model, X_test, y_test, y_pred, y_prob = churn_analysis.build_churn_model(df_processed)
        
        # Deploy to Viya
        churn_analysis.deploy_to_viya()
        
        print("Churn analysis completed successfully!")
        
    finally:
        # Close connections
        viya_conn.close_connections()

if __name__ == "__main__":
    run_churn_analysis()