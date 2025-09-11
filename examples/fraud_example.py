"""
Example usage of Fraud Detection Pipeline
"""

import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import swat
from models.fraud_detection import FraudDetectionPipeline

def run_fraud_detection():
    """Example: Complete fraud detection pipeline"""
    
    # Connect to CAS
    cas = swat.CAS('your-cas-server.com', 5570)
    
    try:
        # Initialize fraud detection pipeline
        fraud_pipeline = FraudDetectionPipeline(cas)
        
        # Load sample data (replace with your actual data)
        sample_data = pd.read_csv('transaction_data.csv')
        
        # Create features and train model
        feature_table = fraud_pipeline.create_fraud_features(sample_data)
        model_result = fraud_pipeline.build_fraud_model(sample_data)
        
        # Set up real-time scoring
        esp_project = fraud_pipeline.process_streaming_data('live_transactions')
        
        print("Fraud detection pipeline deployed successfully")
        
    finally:
        cas.close()

if __name__ == "__main__":
    run_fraud_detection()